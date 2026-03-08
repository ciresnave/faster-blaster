/**
 * @file blis_backend.c
 * @brief Intel blis backend implementation
 * 
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "blis_backend.h"
#include "backend_auto_detect.h"     /* fb_auto_populate_ext_ops       */
#include "sym_tables/sym_tables.h"   /* k_lapacke_symbols, k_cblas_ext */
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

/* BLIS native API types */
#ifndef BLIS_TYPES_DEFINED
#define BLIS_TYPES_DEFINED
/* BLIS dim_t / inc_t must match the library's BLIS_INT_TYPE_SIZE.
 * The AMD AOCL LP64 build (and the standalone BLIS at C:/BLIS) compiles
 * with BLIS_INT_TYPE_SIZE=64, so gint_t = int64_t.
 *
 * On Windows, "long int" is only 32 bits (LLP64 model), which would cause
 * an ABI mismatch: stack-allocated stride arguments read by BLIS as 64-bit
 * would consume two of our 32-bit values, scrambling every subsequent
 * argument (pointers, etc.) and producing a STATUS_ACCESS_VIOLATION crash.
 *
 * Fix: use int64_t unconditionally to match the LP64 BLIS build. */
#include <stdint.h>
typedef int64_t  dim_t;
typedef int64_t  inc_t;
typedef int      conj_t;
typedef int      trans_t;

/* BLIS conjugate constants */
#define BLIS_NO_CONJUGATE     0
#define BLIS_CONJUGATE        1

/* BLIS transpose constants */
#define BLIS_NO_TRANSPOSE     0
#define BLIS_TRANSPOSE        1
#define BLIS_CONJ_NO_TRANSPOSE 2
#define BLIS_CONJ_TRANSPOSE   3

/* BLIS expert interface types (opaque pointers) */
typedef struct cntx_s cntx_t;
typedef struct rntm_s rntm_t;
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

/* CBLAS fallback types for AOCL BLIS */
typedef void (*cblas_saxpy_t)(const int n, const float alpha, const float *x, const int incx, float *y, const int incy);
typedef void (*cblas_daxpy_t)(const int n, const double alpha, const double *x, const int incx, double *y, const int incy);
typedef float (*cblas_sdot_t)(const int n, const float *x, const int incx, const float *y, const int incy);
typedef double (*cblas_ddot_t)(const int n, const double *x, const int incx, const double *y, const int incy);
typedef float (*cblas_snrm2_t)(const int n, const float *x, const int incx);
typedef double (*cblas_dnrm2_t)(const int n, const double *x, const int incx);
typedef float (*cblas_sasum_t)(const int n, const float *x, const int incx);
typedef double (*cblas_dasum_t)(const int n, const double *x, const int incx);
typedef int (*cblas_isamax_t)(const int n, const float *x, const int incx);
typedef int (*cblas_idamax_t)(const int n, const double *x, const int incx);

/* BLIS typed API function pointers */
/* Level 1: note BLIS level 1 functions have 'v' suffix and different signatures */
typedef void (*bli_saxpyv_t)(conj_t conjx, dim_t n, const float* alpha, const float* x, inc_t incx, float* y, inc_t incy);
typedef void (*bli_daxpyv_t)(conj_t conjx, dim_t n, const double* alpha, const double* x, inc_t incx, double* y, inc_t incy);
typedef void (*bli_sdotv_t)(conj_t conjx, conj_t conjy, dim_t n, const float* x, inc_t incx, const float* y, inc_t incy, float* rho);
typedef void (*bli_ddotv_t)(conj_t conjx, conj_t conjy, dim_t n, const double* x, inc_t incx, const double* y, inc_t incy, double* rho);
typedef void (*bli_scopyv_t)(conj_t conjx, dim_t n, const float* x, inc_t incx, float* y, inc_t incy);
typedef void (*bli_dcopyv_t)(conj_t conjx, dim_t n, const double* x, inc_t incx, double* y, inc_t incy);
typedef void (*bli_sscalv_t)(conj_t conjalpha, dim_t n, const float* alpha, float* x, inc_t incx);
typedef void (*bli_dscalv_t)(conj_t conjalpha, dim_t n, const double* alpha, double* x, inc_t incx);
typedef void (*bli_snorm2v_t)(dim_t n, const float* x, inc_t incx, float* norm);
typedef void (*bli_dnorm2v_t)(dim_t n, const double* x, inc_t incx, double* norm);
typedef void (*bli_sswapv_t)(dim_t n, float* x, inc_t incx, float* y, inc_t incy);
typedef void (*bli_dswapv_t)(dim_t n, double* x, inc_t incx, double* y, inc_t incy);
typedef void (*bli_sasumv_t)(dim_t n, const float* x, inc_t incx, float* asum);
typedef void (*bli_dasumv_t)(dim_t n, const double* x, inc_t incx, double* asum);
typedef void (*bli_samaxv_t)(dim_t n, const float* x, inc_t incx, dim_t* index);
typedef void (*bli_damaxv_t)(dim_t n, const double* x, inc_t incx, dim_t* index);

/* Level 2: BLIS expert interface (_ex variants) */
typedef void (*bli_sgemv_ex_t)(trans_t transa, conj_t conjx, dim_t m, dim_t n, 
                               const float* alpha, const float* a, inc_t rsa, inc_t csa,
                               const float* x, inc_t incx, const float* beta, float* y, inc_t incy,
                               const cntx_t* cntx, const rntm_t* rntm);
typedef void (*bli_dgemv_ex_t)(trans_t transa, conj_t conjx, dim_t m, dim_t n,
                               const double* alpha, const double* a, inc_t rsa, inc_t csa,
                               const double* x, inc_t incx, const double* beta, double* y, inc_t incy,
                               const cntx_t* cntx, const rntm_t* rntm);

/* Level 2: BLIS basic interface (fallback if expert not available) */
typedef void (*bli_sgemv_t)(trans_t transa, conj_t conjx, dim_t m, dim_t n, 
                            const float* alpha, const float* a, inc_t rsa, inc_t csa,
                            const float* x, inc_t incx, const float* beta, float* y, inc_t incy);
typedef void (*bli_dgemv_t)(trans_t transa, conj_t conjx, dim_t m, dim_t n,
                            const double* alpha, const double* a, inc_t rsa, inc_t csa,
                            const double* x, inc_t incx, const double* beta, double* y, inc_t incy);

/* Level 3: BLIS expert interface (_ex variants) */
typedef void (*bli_sgemm_ex_t)(trans_t transa, trans_t transb, dim_t m, dim_t n, dim_t k,
                               const float* alpha, const float* a, inc_t rsa, inc_t csa,
                               const float* b, inc_t rsb, inc_t csb,
                               const float* beta, float* c, inc_t rsc, inc_t csc,
                               const cntx_t* cntx, const rntm_t* rntm);
typedef void (*bli_dgemm_ex_t)(trans_t transa, trans_t transb, dim_t m, dim_t n, dim_t k,
                               const double* alpha, const double* a, inc_t rsa, inc_t csa,
                               const double* b, inc_t rsb, inc_t csb,
                               const double* beta, double* c, inc_t rsc, inc_t csc,
                               const cntx_t* cntx, const rntm_t* rntm);

/* Level 3: BLIS basic interface (fallback if expert not available) */
typedef void (*bli_sgemm_t)(trans_t transa, trans_t transb, dim_t m, dim_t n, dim_t k,
                            const float* alpha, const float* a, inc_t rsa, inc_t csa,
                            const float* b, inc_t rsb, inc_t csb,
                            const float* beta, float* c, inc_t rsc, inc_t csc);
typedef void (*bli_dgemm_t)(trans_t transa, trans_t transb, dim_t m, dim_t n, dim_t k,
                            const double* alpha, const double* a, inc_t rsa, inc_t csa,
                            const double* b, inc_t rsb, inc_t csb,
                            const double* beta, double* c, inc_t rsc, inc_t csc);

/* blis-specific functions */
typedef void (*blis_set_num_threads_t)(int num_threads);
typedef int (*blis_get_max_threads_t)(void);
typedef void (*blis_get_version_t)(int* major, int* minor, int* update);
typedef int (*blis_set_threading_layer_t)(int layer);
typedef void (*blis_verbose_t)(int enable);

/* CBLAS Level 2/3 function pointers */
typedef void (*cblas_sgemv_fn)(CBLAS_LAYOUT Order, CBLAS_TRANSPOSE TransA, const int M, const int N,
                                const float alpha, const float* A, const int lda,
                                const float* X, const int incX, const float beta,
                                float* Y, const int incY);
typedef void (*cblas_dgemv_fn)(CBLAS_LAYOUT Order, CBLAS_TRANSPOSE TransA, const int M, const int N,
                                const double alpha, const double* A, const int lda,
                                const double* X, const int incX, const double beta,
                                double* Y, const int incY);
typedef void (*cblas_sgemm_fn)(CBLAS_LAYOUT Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                                const int M, const int N, const int K,
                                const float alpha, const float* A, const int lda,
                                const float* B, const int ldb, const float beta,
                                float* C, const int ldc);
typedef void (*cblas_dgemm_fn)(CBLAS_LAYOUT Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                                const int M, const int N, const int K,
                                const double alpha, const double* A, const int lda,
                                const double* B, const int ldb, const double beta,
                                double* C, const int ldc);

/* Global blis API structure */
static struct {
    fb_lib_handle_t handle;
    bool initialized;
    
    /* blis control functions */
    blis_set_num_threads_t set_num_threads;
    blis_get_max_threads_t get_max_threads;
    blis_get_version_t get_version;
    blis_set_threading_layer_t set_threading_layer;
    blis_verbose_t verbose;
    
    /* CBLAS fallback (for AOCL BLIS) */
    cblas_saxpy_t cblas_saxpy;
    cblas_daxpy_t cblas_daxpy;
    cblas_sdot_t cblas_sdot;
    cblas_ddot_t cblas_ddot;
    cblas_snrm2_t cblas_snrm2;
    cblas_dnrm2_t cblas_dnrm2;
    cblas_sasum_t cblas_sasum;
    cblas_dasum_t cblas_dasum;
    cblas_isamax_t cblas_isamax;
    cblas_idamax_t cblas_idamax;
    
    /* BLIS Typed API Level 1 */
    bli_saxpyv_t saxpyv;
    bli_daxpyv_t daxpyv;
    bli_sdotv_t sdotv;
    bli_ddotv_t ddotv;
    bli_scopyv_t scopyv;
    bli_dcopyv_t dcopyv;
    bli_sscalv_t sscalv;
    bli_dscalv_t dscalv;
    bli_snorm2v_t snorm2v;
    bli_dnorm2v_t dnorm2v;
    bli_sswapv_t sswapv;
    bli_dswapv_t dswapv;
    bli_sasumv_t sasumv;
    bli_dasumv_t dasumv;
    bli_samaxv_t samaxv;
    bli_damaxv_t damaxv;
    
    /* BLIS Typed API Level 2 (expert interface preferred) */
    bli_sgemv_ex_t sgemv_ex;
    bli_dgemv_ex_t dgemv_ex;
    bli_sgemv_t sgemv;  /* basic fallback */
    bli_dgemv_t dgemv;  /* basic fallback */
    
    /* BLIS Typed API Level 3 (expert interface preferred) */
    bli_sgemm_ex_t sgemm_ex;
    bli_dgemm_ex_t dgemm_ex;
    bli_sgemm_t sgemm;  /* basic fallback */
    bli_dgemm_t dgemm;  /* basic fallback */
    
} g_blis;

/* blis threading layer constants */
enum blis_THREADING_LAYER {
    blis_THREADING_INTEL = 0,
    blis_THREADING_SEQUENTIAL = 1,
    blis_THREADING_TBB = 2,
    blis_THREADING_GNU = 3
};

static int load_blis_library(void) {
    if (g_blis.initialized) {
        return 0;
    }
    
    const char* lib_names[] = {
#ifdef _WIN32
        /* Try AMD AOCL BLIS first (more complete than generic BLIS on Windows) */
        "AOCL-LibBlis-Win-MT-dll.dll",
        "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\LP64\\AOCL-LibBlis-Win-MT-dll.dll",
        "AOCLLibBlis-Win-dll.dll",
        "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\LP64\\AOCL-LibBlis-Win-dll.dll",
        "C:\\Program Files\\AMD\\AOCL-Windows\\amd-blis\\lib\\ILP64\\AOCL-LibBlis-Win-dll.dll",
        /* Then try generic BLIS */
        "libblis.4.dll",
        "libblis.dll",
        "blis.dll"
#elif defined(__APPLE__)
        "libblis_rt.dylib",
        "libblis_rt.2.dylib",
        "/opt/intel/oneapi/blis/latest/lib/libblis_rt.dylib"
#else
        "libblis_rt.so",
        "libblis_rt.so.2",
        "/opt/intel/oneapi/blis/latest/lib/intel64/libblis_rt.so"
#endif
    };
    
    for (size_t i = 0; i < sizeof(lib_names) / sizeof(lib_names[0]); i++) {
        g_blis.handle = FB_LOAD_LIBRARY(lib_names[i]);
        if (g_blis.handle) {
            break;
        }
    }
    
    if (!g_blis.handle) {
        return -1;
    }
    
    /* Load blis control functions */
    g_blis.set_num_threads = (blis_set_num_threads_t)
        FB_GET_PROC_ADDRESS(g_blis.handle, "blis_set_num_threads");
    g_blis.get_max_threads = (blis_get_max_threads_t)
        FB_GET_PROC_ADDRESS(g_blis.handle, "blis_get_max_threads");
    g_blis.get_version = (blis_get_version_t)
        FB_GET_PROC_ADDRESS(g_blis.handle, "blis_get_version");
    g_blis.set_threading_layer = (blis_set_threading_layer_t)
        FB_GET_PROC_ADDRESS(g_blis.handle, "blis_set_threading_layer");
    g_blis.verbose = (blis_verbose_t)
        FB_GET_PROC_ADDRESS(g_blis.handle, "blis_verbose");
    
    /* Try CBLAS interface first (for AOCL BLIS compatibility) */
    g_blis.cblas_saxpy = (cblas_saxpy_t)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_saxpy");
    g_blis.cblas_daxpy = (cblas_daxpy_t)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_daxpy");
    g_blis.cblas_sdot = (cblas_sdot_t)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_sdot");
    g_blis.cblas_ddot = (cblas_ddot_t)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_ddot");
    g_blis.cblas_snrm2 = (cblas_snrm2_t)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_snrm2");
    g_blis.cblas_dnrm2 = (cblas_dnrm2_t)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_dnrm2");
    g_blis.cblas_sasum = (cblas_sasum_t)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_sasum");
    g_blis.cblas_dasum = (cblas_dasum_t)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_dasum");
    g_blis.cblas_isamax = (cblas_isamax_t)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_isamax");
    g_blis.cblas_idamax = (cblas_idamax_t)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_idamax");
    
    /* Load BLIS typed API functions (note the 'v' suffix for level 1) */
    g_blis.saxpyv = (bli_saxpyv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_saxpyv");
    g_blis.daxpyv = (bli_daxpyv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_daxpyv");
    g_blis.sdotv = (bli_sdotv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_sdotv");
    g_blis.ddotv = (bli_ddotv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_ddotv");
    g_blis.scopyv = (bli_scopyv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_scopyv");
    g_blis.dcopyv = (bli_dcopyv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_dcopyv");
    g_blis.sscalv = (bli_sscalv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_sscalv");
    g_blis.dscalv = (bli_dscalv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_dscalv");
    g_blis.snorm2v = (bli_snorm2v_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_snorm2v");
    g_blis.dnorm2v = (bli_dnorm2v_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_dnorm2v");
    g_blis.sswapv = (bli_sswapv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_sswapv");
    g_blis.dswapv = (bli_dswapv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_dswapv");
    g_blis.sasumv = (bli_sasumv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_sasumv");
    g_blis.dasumv = (bli_dasumv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_dasumv");
    g_blis.samaxv = (bli_samaxv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_samaxv");
    g_blis.damaxv = (bli_damaxv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_damaxv");
    
    /* Load BLIS expert interface for Level 2/3 (faster than basic, try first) */
    g_blis.sgemv_ex = (bli_sgemv_ex_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_sgemv_ex");
    g_blis.dgemv_ex = (bli_dgemv_ex_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_dgemv_ex");
    g_blis.sgemm_ex = (bli_sgemm_ex_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_sgemm_ex");
    g_blis.dgemm_ex = (bli_dgemm_ex_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_dgemm_ex");
    
    /* Load basic interface as fallback (if expert not available in AOCL) */
    if (!g_blis.sgemv_ex) {
        g_blis.sgemv = (bli_sgemv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_sgemv");
        g_blis.dgemv = (bli_dgemv_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_dgemv");
    }
    if (!g_blis.sgemm_ex) {
        g_blis.sgemm = (bli_sgemm_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_sgemm");
        g_blis.dgemm = (bli_dgemm_t)FB_GET_PROC_ADDRESS(g_blis.handle, "bli_dgemm");
    }
    
    g_blis.initialized = true;
    return 0;
}

bool fb_blis_is_available(void) {
    return load_blis_library() == 0;
}

int fb_blis_init(void) {
    return load_blis_library();
}

void fb_blis_shutdown(void) {
    if (g_blis.initialized && g_blis.handle) {
        FB_FREE_LIBRARY(g_blis.handle);
        g_blis.handle = NULL;
        g_blis.initialized = false;
    }
}

int fb_blis_get_version(int* major, int* minor, int* update) {
    if (!g_blis.initialized || !g_blis.get_version) {
        return -1;
    }
    g_blis.get_version(major, minor, update);
    return 0;
}

void fb_blis_set_num_threads(int num_threads) {
    if (g_blis.set_num_threads) {
        g_blis.set_num_threads(num_threads);
    }
}

int fb_blis_get_num_threads(void) {
    if (g_blis.get_max_threads) {
        return g_blis.get_max_threads();
    }
    return 1;
}

int fb_blis_set_threading_layer(const char* layer) {
    if (!g_blis.set_threading_layer) {
        return -1;
    }
    
    int blis_layer = blis_THREADING_INTEL;
    if (strcmp(layer, "sequential") == 0) {
        blis_layer = blis_THREADING_SEQUENTIAL;
    } else if (strcmp(layer, "tbb") == 0) {
        blis_layer = blis_THREADING_TBB;
    } else if (strcmp(layer, "gnu") == 0) {
        blis_layer = blis_THREADING_GNU;
    }
    
    return g_blis.set_threading_layer(blis_layer);
}

void fb_blis_set_verbose(bool enable) {
    if (g_blis.verbose) {
        g_blis.verbose(enable ? 1 : 0);
    }
}

/* Wrapper functions that convert CBLAS-style calls to BLIS typed API */

static void blis_sasum(int n, const float* x, int incx, float* result) {
    /* Try CBLAS first (for AOCL compatibility) */
    if (g_blis.cblas_sasum) {
        *result = g_blis.cblas_sasum(n, x, incx);
        return;
    }
    
    /* Fall back to BLIS native API */
    if (g_blis.sasumv) {
        g_blis.sasumv((dim_t)n, x, (inc_t)incx, result);
    }
}

static void blis_dasum(int n, const double* x, int incx, double* result) {
    if (g_blis.dasumv) {
        g_blis.dasumv((dim_t)n, x, (inc_t)incx, result);
    }
}

static void blis_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
    /* Try CBLAS first (for AOCL compatibility) */
    if (g_blis.cblas_saxpy) {
        g_blis.cblas_saxpy(n, alpha, x, incx, y, incy);
        return;
    }
    
    /* Fall back to BLIS native API */
    if (g_blis.saxpyv && g_blis.initialized) {
        g_blis.saxpyv(BLIS_NO_CONJUGATE, (dim_t)n, &alpha, x, (inc_t)incx, y, (inc_t)incy);
        return;
    }
    
    /* Backend doesn't support this operation */
    (void)n; (void)alpha; (void)x; (void)incx; (void)y; (void)incy;
}

static void blis_daxpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
    if (g_blis.daxpyv) {
        g_blis.daxpyv(BLIS_NO_CONJUGATE, (dim_t)n, &alpha, x, (inc_t)incx, y, (inc_t)incy);
    } else {
        /* Fallback: do nothing - function not available */
        (void)n; (void)alpha; (void)x; (void)incx; (void)y; (void)incy;
    }
}

static void blis_sdot(int n, const float* x, int incx, const float* y, int incy, float* result) {
    /* Try CBLAS first (for AOCL compatibility) */
    if (g_blis.cblas_sdot) {
        *result = g_blis.cblas_sdot(n, x, incx, y, incy);
        return;
    }
    
    /* Fall back to BLIS native API */
    if (g_blis.sdotv) {
        g_blis.sdotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, (dim_t)n, x, (inc_t)incx, y, (inc_t)incy, result);
    }
}

static void blis_ddot(int n, const double* x, int incx, const double* y, int incy, double* result) {
    if (g_blis.ddotv) {
        g_blis.ddotv(BLIS_NO_CONJUGATE, BLIS_NO_CONJUGATE, (dim_t)n, x, (inc_t)incx, y, (inc_t)incy, result);
    }
}

static void blis_scopy(int n, const float* x, int incx, float* y, int incy) {
    if (g_blis.scopyv) {
        g_blis.scopyv(BLIS_NO_CONJUGATE, (dim_t)n, x, (inc_t)incx, y, (inc_t)incy);
    }
}

static void blis_dcopy(int n, const double* x, int incx, double* y, int incy) {
    if (g_blis.dcopyv) {
        g_blis.dcopyv(BLIS_NO_CONJUGATE, (dim_t)n, x, (inc_t)incx, y, (inc_t)incy);
    }
}

static void blis_sscal(int n, float alpha, float* x, int incx) {
    if (g_blis.sscalv) {
        g_blis.sscalv(BLIS_NO_CONJUGATE, (dim_t)n, &alpha, x, (inc_t)incx);
    }
}

static void blis_dscal(int n, double alpha, double* x, int incx) {
    if (g_blis.dscalv) {
        g_blis.dscalv(BLIS_NO_CONJUGATE, (dim_t)n, &alpha, x, (inc_t)incx);
    }
}

static void blis_snrm2(int n, const float* x, int incx, float* result) {
    /* Try CBLAS first (for AOCL compatibility) */
    if (g_blis.cblas_snrm2) {
        *result = g_blis.cblas_snrm2(n, x, incx);
        return;
    }
    
    /* Fall back to BLIS native API */
    if (g_blis.snorm2v) {
        g_blis.snorm2v((dim_t)n, x, (inc_t)incx, result);
    }
}

static void blis_dnrm2(int n, const double* x, int incx, double* result) {
    if (g_blis.dnorm2v) {
        g_blis.dnorm2v((dim_t)n, x, (inc_t)incx, result);
    }
}

static void blis_sswap(int n, float* x, int incx, float* y, int incy) {
    if (g_blis.sswapv) {
        g_blis.sswapv((dim_t)n, x, (inc_t)incx, y, (inc_t)incy);
    }
}

static void blis_dswap(int n, double* x, int incx, double* y, int incy) {
    if (g_blis.dswapv) {
        g_blis.dswapv((dim_t)n, x, (inc_t)incx, y, (inc_t)incy);
    }
}

static void blis_isamax(int n, const float* x, int incx, int* result) {
    /* Try CBLAS first (for AOCL compatibility) */
    if (g_blis.cblas_isamax) {
        *result = g_blis.cblas_isamax(n, x, incx);
        return;
    }
    
    /* Fall back to BLIS native API */
    if (g_blis.samaxv) {
        dim_t index;
        g_blis.samaxv((dim_t)n, x, (inc_t)incx, &index);
        *result = (int)index;
    }
}

static void blis_idamax(int n, const double* x, int incx, int* result) {
    if (g_blis.damaxv) {
        dim_t index;
        g_blis.damaxv((dim_t)n, x, (inc_t)incx, &index);
        *result = (int)index;
    }
}

/* ============================================================================
 * Vtable Adapter Wrappers
 * 
 * These wrappers adapt from the vtable interface (which returns values)
 * to the BLIS backend implementation (which uses output pointers).
 * ========================================================================= */

static float blis_sdot_wrapper(int n, const float* x, int incx, const float* y, int incy) {
    float result = 0.0f;
    blis_sdot(n, x, incx, y, incy, &result);
    return result;
}

static double blis_ddot_wrapper(int n, const double* x, int incx, const double* y, int incy) {
    double result = 0.0;
    blis_ddot(n, x, incx, y, incy, &result);
    return result;
}

static float blis_snrm2_wrapper(int n, const float* x, int incx) {
    float result = 0.0f;
    blis_snrm2(n, x, incx, &result);
    return result;
}

static double blis_dnrm2_wrapper(int n, const double* x, int incx) {
    double result = 0.0;
    blis_dnrm2(n, x, incx, &result);
    return result;
}

static float blis_sasum_wrapper(int n, const float* x, int incx) {
    float result = 0.0f;
    blis_sasum(n, x, incx, &result);
    return result;
}

static double blis_dasum_wrapper(int n, const double* x, int incx) {
    double result = 0.0;
    blis_dasum(n, x, incx, &result);
    return result;
}

static int blis_isamax_wrapper(int n, const float* x, int incx) {
    int result = 0;
    blis_isamax(n, x, incx, &result);
    return result;
}

static int blis_idamax_wrapper(int n, const double* x, int incx) {
    int result = 0;
    blis_idamax(n, x, incx, &result);
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

static void blis_scasum(int n, const void* x, int incx, float* result) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_scasum");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void blis_dzasum(int n, const void* x, int incx, double* result) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dzasum");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void blis_caxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_caxpy");
    if (fn) {
        fn(n, alpha, x, incx, y, incy);
    }
}

static void blis_zaxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zaxpy");
    if (fn) {
        fn(n, alpha, x, incx, y, incy);
    }
}

static void blis_ccopy(int n, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ccopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void blis_zcopy(int n, const void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zcopy");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void blis_cscal(int n, const void* alpha, void* x, int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_cscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void blis_zscal(int n, const void* alpha, void* x, int incx) {
    typedef void (*fn_t)(int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void blis_csscal(int n, float alpha, void* x, int incx) {
    typedef void (*fn_t)(int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_csscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void blis_zdscal(int n, double alpha, void* x, int incx) {
    typedef void (*fn_t)(int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zdscal");
    if (fn) {
        fn(n, alpha, x, incx);
    }
}

static void blis_cswap(int n, void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_cswap");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void blis_zswap(int n, void* x, int incx, void* y, int incy) {
    typedef void (*fn_t)(int, void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zswap");
    if (fn) {
        fn(n, x, incx, y, incy);
    }
}

static void blis_cdotu_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_cdotu_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void blis_zdotu_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zdotu_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void blis_cdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_cdotc_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void blis_zdotc_sub(int n, const void* x, int incx, const void* y, int incy, void* result) {
    typedef void (*fn_t)(int, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zdotc_sub");
    if (fn) {
        fn(n, x, incx, y, incy, result);
    }
}

static void blis_scnrm2(int n, const void* x, int incx, float* result) {
    typedef float (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_scnrm2");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void blis_dznrm2(int n, const void* x, int incx, double* result) {
    typedef double (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dznrm2");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void blis_icamax(int n, const void* x, int incx, int* result) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_icamax");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

static void blis_izamax(int n, const void* x, int incx, int* result) {
    typedef int (*fn_t)(int, const void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_izamax");
    if (fn) {
        *result = fn(n, x, incx);
    }
}

/* Rotation operations */

static void blis_srotg(float* a, float* b, float* c, float* s) {
    typedef void (*fn_t)(float*, float*, float*, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_srotg");
    if (fn) {
        fn(a, b, c, s);
    }
}

static void blis_drotg(double* a, double* b, double* c, double* s) {
    typedef void (*fn_t)(double*, double*, double*, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_drotg");
    if (fn) {
        fn(a, b, c, s);
    }
}

static void blis_srot(int n, float* x, int incx, float* y, int incy, float c, float s) {
    typedef void (*fn_t)(int, float*, int, float*, int, float, float);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_srot");
    if (fn) {
        fn(n, x, incx, y, incy, c, s);
    } else {
        fprintf(stderr, "BLIS: cblas_srot not available\n");
    }
}

static void blis_drot(int n, double* x, int incx, double* y, int incy, double c, double s) {
    typedef void (*fn_t)(int, double*, int, double*, int, double, double);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_drot");
    if (fn) {
        fn(n, x, incx, y, incy, c, s);
    } else {
        fprintf(stderr, "BLIS: cblas_drot not available\n");
    }
}

static void blis_srotm(int n, float* x, int incx, float* y, int incy, const float* param) {
    typedef void (*fn_t)(int, float*, int, float*, int, const float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_srotm");
    if (fn) {
        fn(n, x, incx, y, incy, param);
    }
}

static void blis_drotm(int n, double* x, int incx, double* y, int incy, const double* param) {
    typedef void (*fn_t)(int, double*, int, double*, int, const double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_drotm");
    if (fn) {
        fn(n, x, incx, y, incy, param);
    }
}

static void blis_srotmg(float* d1, float* d2, float* x1, float y1, float* param) {
    typedef void (*fn_t)(float*, float*, float*, float, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_srotmg");
    if (fn) {
        fn(d1, d2, x1, y1, param);
    }
}

static void blis_drotmg(double* d1, double* d2, double* x1, double y1, double* param) {
    typedef void (*fn_t)(double*, double*, double*, double, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_drotmg");
    if (fn) {
        fn(d1, d2, x1, y1, param);
    }
}

/* Level 2 BLAS operations - Convert CBLAS interface to BLIS typed API */

static void blis_sgemv(const fb_layout_t layout, const fb_transpose_t trans, const int m, const int n, const float alpha,
                      const float* a, const int lda, const float* x, const int incx,
                      const float beta, float* y, const int incy) {
    /* Prefer CBLAS interface: uses plain int parameters with no ABI ambiguity.
     * The BLIS typed API (bli_sgemv_ex / bli_sgemv) uses dim_t/inc_t which are
     * int64_t in the LP64 AOCL build; these are also undeclared in the public
     * AOCL header, so their exact calling convention is uncertain.  CBLAS is
     * the stable, documented, public API. */
    cblas_sgemv_fn cblas_fn = (cblas_sgemv_fn)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_sgemv");
    if (cblas_fn) {
        CBLAS_LAYOUT order = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_TRANSPOSE trans_cblas = (trans == FB_NO_TRANS) ? CblasNoTrans :
                                      (trans == FB_TRANS)    ? CblasTrans    : CblasConjTrans;
        cblas_fn(order, trans_cblas, m, n, alpha, a, lda, x, incx, beta, y, incy);
        return;
    }

    /* BLIS native fallback (if CBLAS not available) */
    inc_t rsa, csa;
    if (layout == FB_LAYOUT_ROW_MAJOR) {
        rsa = (inc_t)lda;  csa = 1;
    } else {
        rsa = 1;  csa = (inc_t)lda;
    }
    trans_t blis_trans = (trans == FB_NO_TRANS) ? BLIS_NO_TRANSPOSE :
                         (trans == FB_TRANS) ? BLIS_TRANSPOSE : BLIS_CONJ_TRANSPOSE;
    float alpha_val = alpha;
    float beta_val = beta;
    if (g_blis.sgemv_ex) {
        g_blis.sgemv_ex(blis_trans, BLIS_NO_CONJUGATE,
                        (dim_t)m, (dim_t)n, &alpha_val, a, rsa, csa,
                        x, (inc_t)incx, &beta_val, y, (inc_t)incy, NULL, NULL);
    } else if (g_blis.sgemv) {
        g_blis.sgemv(blis_trans, BLIS_NO_CONJUGATE,
                     (dim_t)m, (dim_t)n, &alpha_val, a, rsa, csa,
                     x, (inc_t)incx, &beta_val, y, (inc_t)incy);
    }
}

static void blis_dgemv(const fb_layout_t layout, const fb_transpose_t trans, const int m, const int n, const double alpha,
                      const double* a, const int lda, const double* x, const int incx,
                      const double beta, double* y, const int incy) {
    cblas_dgemv_fn cblas_fn = (cblas_dgemv_fn)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_dgemv");
    if (cblas_fn) {
        CBLAS_LAYOUT order = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_TRANSPOSE trans_cblas = (trans == FB_NO_TRANS) ? CblasNoTrans :
                                      (trans == FB_TRANS)    ? CblasTrans    : CblasConjTrans;
        cblas_fn(order, trans_cblas, m, n, alpha, a, lda, x, incx, beta, y, incy);
        return;
    }
    inc_t rsa, csa;
    if (layout == FB_LAYOUT_ROW_MAJOR) { rsa = (inc_t)lda; csa = 1; }
    else                               { rsa = 1; csa = (inc_t)lda; }
    trans_t blis_trans = (trans == FB_NO_TRANS) ? BLIS_NO_TRANSPOSE :
                         (trans == FB_TRANS) ? BLIS_TRANSPOSE : BLIS_CONJ_TRANSPOSE;
    double alpha_val = alpha, beta_val = beta;
    if (g_blis.dgemv_ex) {
        g_blis.dgemv_ex(blis_trans, BLIS_NO_CONJUGATE,
                        (dim_t)m, (dim_t)n, &alpha_val, a, rsa, csa,
                        x, (inc_t)incx, &beta_val, y, (inc_t)incy, NULL, NULL);
    } else if (g_blis.dgemv) {
        g_blis.dgemv(blis_trans, BLIS_NO_CONJUGATE,
                     (dim_t)m, (dim_t)n, &alpha_val, a, rsa, csa,
                     x, (inc_t)incx, &beta_val, y, (inc_t)incy);
    }
}

static void blis_sgemm(const fb_layout_t layout, const fb_transpose_t transa, const fb_transpose_t transb, const int m, const int n, const int k,
                      const float alpha, const float* a, const int lda,
                      const float* b, const int ldb, const float beta,
                      float* c, const int ldc) {
    /* Convert layout + lda/ldb/ldc to BLIS row/column strides */
    inc_t rsa, csa, rsb, csb, rsc, csc;
    if (layout == FB_LAYOUT_ROW_MAJOR) {
        rsa = (inc_t)lda; csa = 1;
        rsb = (inc_t)ldb; csb = 1;
        rsc = (inc_t)ldc; csc = 1;
    } else {
        rsa = 1; csa = (inc_t)lda;
        rsb = 1; csb = (inc_t)ldb;
        rsc = 1; csc = (inc_t)ldc;
    }
    
    /* Convert transpose flags */
    trans_t blis_transa = (transa == FB_NO_TRANS) ? BLIS_NO_TRANSPOSE :
                          (transa == FB_TRANS) ? BLIS_TRANSPOSE : BLIS_CONJ_TRANSPOSE;
    trans_t blis_transb = (transb == FB_NO_TRANS) ? BLIS_NO_TRANSPOSE :
                          (transb == FB_TRANS) ? BLIS_TRANSPOSE : BLIS_CONJ_TRANSPOSE;
    
    /* BLIS uses pointer arguments for alpha/beta */
    float alpha_val = alpha;
    float beta_val = beta;
    
    /* Prefer CBLAS interface (stable API, unambiguous int parameters) */
    cblas_sgemm_fn cblas_fn = (cblas_sgemm_fn)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_sgemm");
    if (cblas_fn) {
        CBLAS_LAYOUT order = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_TRANSPOSE ta = (transa == FB_NO_TRANS) ? CblasNoTrans : (transa == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_TRANSPOSE tb = (transb == FB_NO_TRANS) ? CblasNoTrans : (transb == FB_TRANS) ? CblasTrans : CblasConjTrans;
        cblas_fn(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }
    /* BLIS native fallback */
    if (g_blis.sgemm_ex) {
        g_blis.sgemm_ex(blis_transa, blis_transb,
                        (dim_t)m, (dim_t)n, (dim_t)k,
                        &alpha_val, a, rsa, csa,
                        b, rsb, csb,
                        &beta_val, c, rsc, csc,
                        NULL, NULL);
    } else if (g_blis.sgemm) {
        g_blis.sgemm(blis_transa, blis_transb,
                     (dim_t)m, (dim_t)n, (dim_t)k,
                     &alpha_val, a, rsa, csa,
                     b, rsb, csb,
                     &beta_val, c, rsc, csc);
    }
}

static void blis_dgemm(const fb_layout_t layout, const fb_transpose_t transa, const fb_transpose_t transb, const int m, const int n, const int k,
                      const double alpha, const double* a, const int lda,
                      const double* b, const int ldb, const double beta,
                      double* c, const int ldc) {
    /* Convert layout + lda/ldb/ldc to BLIS row/column strides */
    inc_t rsa, csa, rsb, csb, rsc, csc;
    if (layout == FB_LAYOUT_ROW_MAJOR) {
        rsa = (inc_t)lda; csa = 1;
        rsb = (inc_t)ldb; csb = 1;
        rsc = (inc_t)ldc; csc = 1;
    } else {
        rsa = 1; csa = (inc_t)lda;
        rsb = 1; csb = (inc_t)ldb;
        rsc = 1; csc = (inc_t)ldc;
    }
    
    /* Convert transpose flags */
    trans_t blis_transa = (transa == FB_NO_TRANS) ? BLIS_NO_TRANSPOSE :
                          (transa == FB_TRANS) ? BLIS_TRANSPOSE : BLIS_CONJ_TRANSPOSE;
    trans_t blis_transb = (transb == FB_NO_TRANS) ? BLIS_NO_TRANSPOSE :
                          (transb == FB_TRANS) ? BLIS_TRANSPOSE : BLIS_CONJ_TRANSPOSE;
    
    /* BLIS uses pointer arguments for alpha/beta */
    double alpha_val = alpha;
    double beta_val = beta;
    
    /* Prefer CBLAS interface (stable API, unambiguous int parameters) */
    cblas_dgemm_fn cblas_fn = (cblas_dgemm_fn)FB_GET_PROC_ADDRESS(g_blis.handle, "cblas_dgemm");
    if (cblas_fn) {
        CBLAS_LAYOUT order = (layout == FB_LAYOUT_ROW_MAJOR) ? CblasRowMajor : CblasColMajor;
        CBLAS_TRANSPOSE ta = (transa == FB_NO_TRANS) ? CblasNoTrans : (transa == FB_TRANS) ? CblasTrans : CblasConjTrans;
        CBLAS_TRANSPOSE tb = (transb == FB_NO_TRANS) ? CblasNoTrans : (transb == FB_TRANS) ? CblasTrans : CblasConjTrans;
        cblas_fn(order, ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }
    /* BLIS native fallback */
    if (g_blis.dgemm_ex) {
        g_blis.dgemm_ex(blis_transa, blis_transb,
                        (dim_t)m, (dim_t)n, (dim_t)k,
                        &alpha_val, a, rsa, csa,
                        b, rsb, csb,
                        &beta_val, c, rsc, csc,
                        NULL, NULL);
    } else if (g_blis.dgemm) {
        g_blis.dgemm(blis_transa, blis_transb,
                     (dim_t)m, (dim_t)n, (dim_t)k,
                     &alpha_val, a, rsa, csa,
                     b, rsb, csb,
                     &beta_val, c, rsc, csc);
    }
}

/* Level 2 BLAS - Complex GEMV */
static void blis_cgemv(const fb_layout_t layout, const fb_transpose_t trans, int m, int n, const fb_complex_float_t alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_cgemv");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_TRANSPOSE)(int)trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy); }
}

static void blis_zgemv(const fb_layout_t layout, const fb_transpose_t trans, int m, int n, const fb_complex_double_t alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zgemv");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_TRANSPOSE)(int)trans, m, n, &alpha, a, lda, x, incx, &beta, y, incy); }
}

/* Level 2 BLAS - GBMV (banded matrix-vector) */
static void blis_sgbmv(const fb_layout_t layout, const fb_transpose_t trans, const int m, const int n, const int kl, const int ku, const float alpha, const float* a, const int lda,
                      const float* x, const int incx, const float beta, float* y, const int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_sgbmv");
    if (fn) { fn((int)layout, (int)trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void blis_dgbmv(const fb_layout_t layout, const fb_transpose_t trans, const int m, const int n, const int kl, const int ku, const double alpha, const double* a, const int lda,
                      const double* x, const int incx, const double beta, double* y, const int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dgbmv");
    if (fn) { fn((int)layout, (int)trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void blis_cgbmv(const fb_layout_t layout, const fb_transpose_t trans, int m, int n, int kl, int ku, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_cgbmv");
    if (fn) { fn((int)layout, (int)trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

static void blis_zgbmv(const fb_layout_t layout, const fb_transpose_t trans, int m, int n, int kl, int ku, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, int, int, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zgbmv");
    if (fn) { fn((int)layout, (int)trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HEMV (Hermitian matrix-vector) */
static void blis_chemv(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_chemv");
    if (fn) { fn((int)layout, (int)uplo, n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void blis_zhemv(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zhemv");
    if (fn) { fn((int)layout, (int)uplo, n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HBMV (Hermitian banded matrix-vector) */
static void blis_chbmv(const fb_layout_t layout, const fb_uplo_t uplo, int n, int k, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_chbmv");
    if (fn) { fn((int)layout, (int)uplo, n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

static void blis_zhbmv(const fb_layout_t layout, const fb_uplo_t uplo, int n, int k, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zhbmv");
    if (fn) { fn((int)layout, (int)uplo, n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - HPMV (Hermitian packed matrix-vector) */
static void blis_chpmv(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* ap,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_chpmv");
    if (fn) { fn((int)layout, (int)uplo, n, alpha, ap, x, incx, beta, y, incy); }
}

static void blis_zhpmv(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* ap,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zhpmv");
    if (fn) { fn((int)layout, (int)uplo, n, alpha, ap, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SYMV (symmetric matrix-vector) */
static void blis_ssymv(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const float alpha, const float* a, const int lda,
                      const float* x, const int incx, const float beta, float* y, const int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ssymv");
    if (fn) { fn((int)layout, (int)uplo, n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void blis_dsymv(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const double alpha, const double* a, const int lda,
                      const double* x, const int incx, const double beta, double* y, const int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dsymv");
    if (fn) { fn((int)layout, (int)uplo, n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void blis_csymv(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* a, int lda,
                      const fb_complex_float_t* x, int incx, const fb_complex_float_t* beta, fb_complex_float_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_csymv");
    if (fn) { fn((int)layout, (int)uplo, n, alpha, a, lda, x, incx, beta, y, incy); }
}

static void blis_zsymv(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* a, int lda,
                      const fb_complex_double_t* x, int incx, const fb_complex_double_t* beta, fb_complex_double_t* y, int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zsymv");
    if (fn) { fn((int)layout, (int)uplo, n, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SBMV (symmetric banded matrix-vector) */
static void blis_ssbmv(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const int k, const float alpha, const float* a, const int lda,
                      const float* x, const int incx, const float beta, float* y, const int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ssbmv");
    if (fn) { fn((int)layout, (int)uplo, n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

static void blis_dsbmv(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const int k, const double alpha, const double* a, const int lda,
                      const double* x, const int incx, const double beta, double* y, const int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dsbmv");
    if (fn) { fn((int)layout, (int)uplo, n, k, alpha, a, lda, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - SPMV (symmetric packed matrix-vector) */
static void blis_sspmv(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const float alpha, const float* ap,
                      const float* x, const int incx, const float beta, float* y, const int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_sspmv");
    if (fn) { fn((int)layout, (int)uplo, n, alpha, ap, x, incx, beta, y, incy); }
}

static void blis_dspmv(const fb_layout_t layout, const fb_uplo_t uplo, const int n, const double alpha, const double* ap,
                      const double* x, const int incx, const double beta, double* y, const int incy) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dspmv");
    if (fn) { fn((int)layout, (int)uplo, n, alpha, ap, x, incx, beta, y, incy); }
}

/* Level 2 BLAS - TRMV (triangular matrix-vector) */
static void blis_strmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, const int n, const float* a, const int lda, float* x, const int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_strmv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx); }
}

static void blis_dtrmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, const int n, const double* a, const int lda, double* x, const int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dtrmv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx); }
}

static void blis_ctrmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_float_t* a, int lda, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ctrmv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx); }
}

static void blis_ztrmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_double_t* a, int lda, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ztrmv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx); }
}

/* Level 2 BLAS - TBMV (triangular banded matrix-vector) */
static void blis_stbmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, const int n, const int k, const float* a, const int lda, float* x, const int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_stbmv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx); }
}

static void blis_dtbmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, const int n, const int k, const double* a, const int lda, double* x, const int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dtbmv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx); }
}

static void blis_ctbmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, int k, const fb_complex_float_t* a, int lda, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ctbmv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx); }
}

static void blis_ztbmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, int k, const fb_complex_double_t* a, int lda, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ztbmv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx); }
}

/* Level 2 BLAS - TPMV (triangular packed matrix-vector) */
static void blis_stpmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, const int n, const float* ap, float* x, const int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_stpmv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx); }
}

static void blis_dtpmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, const int n, const double* ap, double* x, const int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dtpmv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx); }
}

static void blis_ctpmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_float_t* ap, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ctpmv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx); }
}

static void blis_ztpmv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_double_t* ap, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ztpmv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx); }
}

/* Level 2 BLAS - TRSV (triangular solve) */
static void blis_strsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, const int n, const float* a, const int lda, float* x, const int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_strsv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx); }
}

static void blis_dtrsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, const int n, const double* a, const int lda, double* x, const int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dtrsv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx); }
}

static void blis_ctrsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_float_t* a, int lda, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ctrsv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx); }
}

static void blis_ztrsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_double_t* a, int lda, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ztrsv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, a, lda, x, incx); }
}

/* Level 2 BLAS - TBSV (triangular banded solve) */
static void blis_stbsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, const int n, const int k, const float* a, const int lda, float* x, const int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_stbsv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx); }
}

static void blis_dtbsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, const int n, const int k, const double* a, const int lda, double* x, const int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dtbsv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx); }
}

static void blis_ctbsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, int k, const fb_complex_float_t* a, int lda, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ctbsv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx); }
}

static void blis_ztbsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, int k, const fb_complex_double_t* a, int lda, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ztbsv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, k, a, lda, x, incx); }
}

/* Level 2 BLAS - TPSV (triangular packed solve) */
static void blis_stpsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, const int n, const float* ap, float* x, const int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const float*, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_stpsv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx); }
}

static void blis_dtpsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, const int n, const double* ap, double* x, const int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const double*, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dtpsv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx); }
}

static void blis_ctpsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_float_t* ap, fb_complex_float_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ctpsv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx); }
}

static void blis_ztpsv(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, const fb_diag_t diag, int n, const fb_complex_double_t* ap, fb_complex_double_t* x, int incx) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ztpsv");
    if (fn) { fn((int)layout, (int)uplo, (int)trans, (int)diag, n, ap, x, incx); }
}

/* Level 2 BLAS - GER (rank-1 update) */
static void blis_sger(const fb_layout_t layout, int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, float, const float*, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_sger");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void blis_dger(const fb_layout_t layout, int m, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, double, const double*, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dger");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void blis_cgeru(int m, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx,
                      const fb_complex_float_t* y, int incy, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_cgeru");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void blis_zgeru(int m, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx,
                      const fb_complex_double_t* y, int incy, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zgeru");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void blis_cgerc(int m, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx,
                      const fb_complex_float_t* y, int incy, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_cgerc");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

static void blis_zgerc(int m, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx,
                      const fb_complex_double_t* y, int incy, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, int, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zgerc");
    if (fn) { fn(CblasColMajor, m, n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - HER (Hermitian rank-1 update) */
static void blis_cher(const fb_layout_t layout, const fb_uplo_t uplo, int n, float alpha, const fb_complex_float_t* x, int incx, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_cher");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, a, lda); }
}

static void blis_zher(const fb_layout_t layout, const fb_uplo_t uplo, int n, double alpha, const fb_complex_double_t* x, int incx, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zher");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, a, lda); }
}

/* Level 2 BLAS - HPR (Hermitian packed rank-1 update) */
static void blis_chpr(const fb_layout_t layout, const fb_uplo_t uplo, int n, float alpha, const fb_complex_float_t* x, int incx, fb_complex_float_t* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_chpr");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, ap); }
}

static void blis_zhpr(const fb_layout_t layout, const fb_uplo_t uplo, int n, double alpha, const fb_complex_double_t* x, int incx, fb_complex_double_t* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zhpr");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, ap); }
}

/* Level 2 BLAS - HER2 (Hermitian rank-2 update) */
static void blis_cher2(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx,
                      const fb_complex_float_t* y, int incy, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_cher2");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, y, incy, a, lda); }
}

static void blis_zher2(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx,
                      const fb_complex_double_t* y, int incy, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zher2");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - HPR2 (Hermitian packed rank-2 update) */
static void blis_chpr2(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx,
                      const fb_complex_float_t* y, int incy, fb_complex_float_t* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_chpr2");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, y, incy, ap); }
}

static void blis_zhpr2(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx,
                      const fb_complex_double_t* y, int incy, fb_complex_double_t* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, const void*, int, void*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zhpr2");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, y, incy, ap); }
}

/* Level 2 BLAS - SYR (symmetric rank-1 update) */
static void blis_ssyr(const fb_layout_t layout, const fb_uplo_t uplo, int n, float alpha, const float* x, int incx, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ssyr");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, a, lda); }
}

static void blis_dsyr(const fb_layout_t layout, const fb_uplo_t uplo, int n, double alpha, const double* x, int incx, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dsyr");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, a, lda); }
}

static void blis_csyr(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_float_t* alpha, const fb_complex_float_t* x, int incx, fb_complex_float_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_csyr");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, a, lda); }
}

static void blis_zsyr(const fb_layout_t layout, const fb_uplo_t uplo, int n, const fb_complex_double_t* alpha, const fb_complex_double_t* x, int incx, fb_complex_double_t* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zsyr");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, a, lda); }
}

/* Level 2 BLAS - SPR (symmetric packed rank-1 update) */
static void blis_sspr(const fb_layout_t layout, const fb_uplo_t uplo, int n, float alpha, const float* x, int incx, float* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_sspr");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, ap); }
}

static void blis_dspr(const fb_layout_t layout, const fb_uplo_t uplo, int n, double alpha, const double* x, int incx, double* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dspr");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, ap); }
}

/* Level 2 BLAS - SYR2 (symmetric rank-2 update) */
static void blis_ssyr2(const fb_layout_t layout, const fb_uplo_t uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ssyr2");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, y, incy, a, lda); }
}

static void blis_dsyr2(const fb_layout_t layout, const fb_uplo_t uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* a, int lda) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dsyr2");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, y, incy, a, lda); }
}

/* Level 2 BLAS - SPR2 (symmetric packed rank-2 update) */
static void blis_sspr2(const fb_layout_t layout, const fb_uplo_t uplo, int n, float alpha, const float* x, int incx, const float* y, int incy, float* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, float, const float*, int, const float*, int, float*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_sspr2");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, y, incy, ap); }
}

static void blis_dspr2(const fb_layout_t layout, const fb_uplo_t uplo, int n, double alpha, const double* x, int incx, const double* y, int incy, double* ap) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, int, double, const double*, int, const double*, int, double*);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dspr2");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, n, alpha, x, incx, y, incy, ap); }
}

/* Level 3 BLAS - GEMM (complex) */
static void blis_cgemm(const fb_layout_t layout, const fb_transpose_t transa, const fb_transpose_t transb, int m, int n, int k, const fb_complex_float_t alpha,
                      const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                      const fb_complex_float_t beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
                         const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_cgemm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_TRANSPOSE)(int)transa, (CBLAS_TRANSPOSE)(int)transb,
                 m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc); }
}

static void blis_zgemm(const fb_layout_t layout, const fb_transpose_t transa, const fb_transpose_t transb, int m, int n, int k, const fb_complex_double_t alpha,
                      const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                      const fb_complex_double_t beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_TRANSPOSE, CBLAS_TRANSPOSE, int, int, int,
                         const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zgemm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_TRANSPOSE)(int)transa, (CBLAS_TRANSPOSE)(int)transb,
                 m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc); }
}

/* Level 3 BLAS - SYMM (symmetric matrix-matrix) */
static void blis_ssymm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, int m, int n, float alpha, const float* a, int lda,
                      const float* b, int ldb, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, float,
                         const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ssymm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void blis_dsymm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, int m, int n, double alpha, const double* a, int lda,
                      const double* b, int ldb, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, double,
                         const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dsymm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void blis_csymm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, int m, int n, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                      const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_csymm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void blis_zsymm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, int m, int n, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                      const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zsymm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - HEMM (Hermitian matrix-matrix) */
static void blis_chemm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, int m, int n, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                      const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_chemm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void blis_zhemm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, int m, int n, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                      const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, int, int, const void*,
                         const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zhemm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - SYRK (symmetric rank-k update) */
static void blis_ssyrk(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, float alpha, const float* a, int lda, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ssyrk");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)trans, n, k, alpha, a, lda, beta, c, ldc); }
}

static void blis_dsyrk(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, double alpha, const double* a, int lda, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dsyrk");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)trans, n, k, alpha, a, lda, beta, c, ldc); }
}

static void blis_csyrk(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_csyrk");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)trans, n, k, alpha, a, lda, beta, c, ldc); }
}

static void blis_zsyrk(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zsyrk");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)trans, n, k, alpha, a, lda, beta, c, ldc); }
}

/* Level 3 BLAS - HERK (Hermitian rank-k update) */
static void blis_cherk(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, float alpha, const fb_complex_float_t* a, int lda, float beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_cherk");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)trans, n, k, alpha, a, lda, beta, c, ldc); }
}

static void blis_zherk(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, double alpha, const fb_complex_double_t* a, int lda, double beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zherk");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)trans, n, k, alpha, a, lda, beta, c, ldc); }
}

/* Level 3 BLAS - SYR2K (symmetric rank-2k update) */
static void blis_ssyr2k(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, float alpha, const float* a, int lda,
                       const float* b, int ldb, float beta, float* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, float, const float*, int, const float*, int, float, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ssyr2k");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void blis_dsyr2k(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, double alpha, const double* a, int lda,
                       const double* b, int ldb, double beta, double* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, double, const double*, int, const double*, int, double, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dsyr2k");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void blis_csyr2k(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, const fb_complex_float_t* alpha,
                       const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                       const fb_complex_float_t* beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_csyr2k");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void blis_zsyr2k(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, const fb_complex_double_t* alpha,
                       const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                       const fb_complex_double_t* beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, const void*, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zsyr2k");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - HER2K (Hermitian rank-2k update) */
static void blis_cher2k(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, const fb_complex_float_t* alpha,
                       const fb_complex_float_t* a, int lda, const fb_complex_float_t* b, int ldb,
                       float beta, fb_complex_float_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, float, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_cher2k");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

static void blis_zher2k(const fb_layout_t layout, const fb_uplo_t uplo, const fb_transpose_t trans, int n, int k, const fb_complex_double_t* alpha,
                       const fb_complex_double_t* a, int lda, const fb_complex_double_t* b, int ldb,
                       double beta, fb_complex_double_t* c, int ldc) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_UPLO, CBLAS_TRANSPOSE, int, int, const void*, const void*, int, const void*, int, double, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_zher2k");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }
}

/* Level 3 BLAS - TRMM (triangular matrix-matrix) */
static void blis_strmm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n, float alpha, const float* a, int lda, float* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_strmm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)transa, (CBLAS_DIAG)(int)diag, m, n, alpha, a, lda, b, ldb); }
}

static void blis_dtrmm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n, double alpha, const double* a, int lda, double* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dtrmm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)transa, (CBLAS_DIAG)(int)diag, m, n, alpha, a, lda, b, ldb); }
}

static void blis_ctrmm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, fb_complex_float_t* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ctrmm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)transa, (CBLAS_DIAG)(int)diag, m, n, alpha, a, lda, b, ldb); }
}

static void blis_ztrmm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, fb_complex_double_t* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ztrmm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)transa, (CBLAS_DIAG)(int)diag, m, n, alpha, a, lda, b, ldb); }
}

/* Level 3 BLAS - TRSM (triangular solve matrix) */
static void blis_strsm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n, float alpha, const float* a, int lda, float* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, float, const float*, int, float*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_strsm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)transa, (CBLAS_DIAG)(int)diag, m, n, alpha, a, lda, b, ldb); }
}

static void blis_dtrsm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n, double alpha, const double* a, int lda, double* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, double, const double*, int, double*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_dtrsm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)transa, (CBLAS_DIAG)(int)diag, m, n, alpha, a, lda, b, ldb); }
}

static void blis_ctrsm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n, const fb_complex_float_t* alpha,
                      const fb_complex_float_t* a, int lda, fb_complex_float_t* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ctrsm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)transa, (CBLAS_DIAG)(int)diag, m, n, alpha, a, lda, b, ldb); }
}

static void blis_ztrsm(const fb_layout_t layout, const fb_side_t side, const fb_uplo_t uplo, const fb_transpose_t transa, const fb_diag_t diag, int m, int n, const fb_complex_double_t* alpha,
                      const fb_complex_double_t* a, int lda, fb_complex_double_t* b, int ldb) {
    typedef void (*fn_t)(CBLAS_LAYOUT, CBLAS_SIDE, CBLAS_UPLO, CBLAS_TRANSPOSE, CBLAS_DIAG, int, int, const void*, const void*, int, void*, int);
    fn_t fn = (fn_t)FB_LOAD_SYMBOL(g_blis.handle, "cblas_ztrsm");
    if (fn) { fn((CBLAS_LAYOUT)(int)layout, (CBLAS_SIDE)(int)side, (CBLAS_UPLO)(int)uplo, (CBLAS_TRANSPOSE)(int)transa, (CBLAS_DIAG)(int)diag, m, n, alpha, a, lda, b, ldb); }
}

/* Backend property functions for unified interface */

static uint32_t blis_get_capabilities(void* handle) {
    (void)handle;
    return FB_CAP_CPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3 |
           FB_CAP_SINGLE | FB_CAP_DOUBLE | FB_CAP_COMPLEX;
}

static int blis_get_num_threads_vtable(void* handle) {
    (void)handle;
    return 1;  /* BLIS backend doesn't expose thread control via CBLAS */
}

static void blis_set_num_threads_vtable(void* handle, int num_threads) {
    (void)handle;
    (void)num_threads;
    /* No-op: BLIS thread control not exposed through this backend */
}

/* blis vtable */
static fb_backend_vtable_t g_blis_vtable = {
    /* Level 1 BLAS */
    .sasum = blis_sasum_wrapper,
    .dasum = blis_dasum_wrapper,
    .saxpy = blis_saxpy,
    .daxpy = blis_daxpy,
    .sdot = blis_sdot_wrapper,
    .ddot = blis_ddot_wrapper,
    .scopy = blis_scopy,
    .dcopy = blis_dcopy,
    .sscal = blis_sscal,
    .dscal = blis_dscal,
    .snrm2 = blis_snrm2_wrapper,
    .dnrm2 = blis_dnrm2_wrapper,
    .sswap = blis_sswap,
    .dswap = blis_dswap,
    .isamax = blis_isamax_wrapper,
    .idamax = blis_idamax_wrapper,
    /* NOTE: Rotation operations (srot, drot, srotg, etc.) not included
     * BLIS does not export these as standard CBLAS functions,
     * fall through to AOCL or other backends that support them */
    
    /* Level 2 BLAS */
    .sgemv = blis_sgemv,
    .dgemv = blis_dgemv,
    .sgbmv = blis_sgbmv,
    .dgbmv = blis_dgbmv,
    .ssymv = blis_ssymv,
    .dsymv = blis_dsymv,
    .ssbmv = blis_ssbmv,
    .dsbmv = blis_dsbmv,
    .sspmv = blis_sspmv,
    .dspmv = blis_dspmv,
    .strmv = blis_strmv,
    .dtrmv = blis_dtrmv,
    .stbmv = blis_stbmv,
    .dtbmv = blis_dtbmv,
    .stpmv = blis_stpmv,
    .dtpmv = blis_dtpmv,
    .strsv = blis_strsv,
    .dtrsv = blis_dtrsv,
    .stbsv = blis_stbsv,
    .dtbsv = blis_dtbsv,
    .stpsv = blis_stpsv,
    .dtpsv = blis_dtpsv,
    .sger = blis_sger,
    .dger = blis_dger,
    .ssyr = blis_ssyr,
    .dsyr = blis_dsyr,
    .sspr = blis_sspr,
    .dspr = blis_dspr,
    .ssyr2 = blis_ssyr2,
    .dsyr2 = blis_dsyr2,
    .sspr2 = blis_sspr2,
    .dspr2 = blis_dspr2,
    
    /* Level 3 BLAS */
    .sgemm = blis_sgemm,
    .dgemm = blis_dgemm,
    .ssymm = blis_ssymm,
    .dsymm = blis_dsymm,
    .strmm = blis_strmm,
    .dtrmm = blis_dtrmm,
    .strsm = blis_strsm,
    .dtrsm = blis_dtrsm,
    .ssyrk = blis_ssyrk,
    .dsyrk = blis_dsyrk,
    .ssyr2k = blis_ssyr2k,
    .dsyr2k = blis_dsyr2k,
    
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
    .get_capabilities = blis_get_capabilities,
    .get_num_threads = blis_get_num_threads_vtable,
    .set_num_threads = blis_set_num_threads_vtable
};

const fb_backend_vtable_t* fb_blis_get_vtable(void) {
    if (!fb_blis_is_available()) {
        return NULL;
    }
    /* Lazily populate ext_ops[] via dlsym for any CBLAS/LAPACK symbols BLIS exports.
     * BLIS is primarily BLAS-only, so most LAPACK slots will resolve to NULL.
     * Only NULL slots are touched — typed wrappers (BLAS L1/L2/L3) win. */
    static bool g_ext_ops_populated = false;
    if (!g_ext_ops_populated) {
        fb_auto_populate_ext_ops(&g_blis_vtable, g_blis.handle,
                                 k_lapacke_symbols, k_lapacke_symbols_count);
        fb_auto_populate_ext_ops(&g_blis_vtable, g_blis.handle,
                                 k_cblas_ext_symbols, k_cblas_ext_symbols_count);
        g_ext_ops_populated = true;
    }
    return &g_blis_vtable;
}
