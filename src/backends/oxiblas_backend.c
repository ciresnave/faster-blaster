/**
 * @file oxiblas_backend.c
 * @brief OxiBLAS backend adapter implementation
 *
 * Minimal runtime adapter that dynamically loads liboxiblas_ffi and exposes
 * a CBLAS-convention vtable by wrapping the Fortran-convention symbols
 * (saxpy_, dgemm_, …) that OxiBLAS exports.
 *
 * Only the operations exercised by the test suite are wrapped here.
 * For full operation coverage use the plugin architecture:
 *     src/plugins/plugin_oxiblas.c → fb_enumerate_and_populate()
 *
 * @copyright Copyright (c) 2025
 * @license   MIT OR Apache-2.0
 */

#include "oxiblas_backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/* =========================================================================
 * Platform-portable loader macros
 * ======================================================================== */
#ifdef _WIN32
#  include <windows.h>
#  define FB_LOAD_LIBRARY(p)           LoadLibraryA(p)
#  define FB_GET_PROC_ADDRESS(h, n)    GetProcAddress((HMODULE)(h), n)
#  define FB_FREE_LIBRARY(h)           FreeLibrary((HMODULE)(h))
   typedef HMODULE fb_lib_handle_t;
#else
#  include <dlfcn.h>
#  define FB_LOAD_LIBRARY(p)           dlopen((p), RTLD_LAZY | RTLD_LOCAL)
#  define FB_GET_PROC_ADDRESS(h, n)    dlsym((h), (n))
#  define FB_FREE_LIBRARY(h)           dlclose(h)
   typedef void *fb_lib_handle_t;
#endif

/* =========================================================================
 * Fortran function pointer types
 * Fortran ABI: all arguments passed by pointer.
 * ======================================================================== */
typedef void (*saxpy_fortran_t)(int*, float*, const float*, int*, float*, int*);
typedef void (*daxpy_fortran_t)(int*, double*, const double*, int*, double*, int*);
typedef void (*sgemm_fortran_t)(const char*, const char*,
                                int*, int*, int*,
                                const float*, const float*, int*,
                                const float*, int*,
                                const float*, float*, int*);
typedef void (*dgemm_fortran_t)(const char*, const char*,
                                int*, int*, int*,
                                const double*, const double*, int*,
                                const double*, int*,
                                const double*, double*, int*);

/* =========================================================================
 * Global adapter state
 * ======================================================================== */
static struct {
    fb_lib_handle_t   handle;
    bool              initialized;

    saxpy_fortran_t   saxpy_f;
    daxpy_fortran_t   daxpy_f;
    sgemm_fortran_t   sgemm_f;
    dgemm_fortran_t   dgemm_f;
} g_oxiblas = {0};

static fb_backend_vtable_t g_oxiblas_vtable = {0};

/* Candidate library names */
static const char *const k_names[] = {
#if defined(_WIN32)
    "oxiblas_ffi.dll", "liboxiblas_ffi.dll",
#elif defined(__APPLE__)
    "liboxiblas_ffi.dylib", "liboxiblas_ffi.0.dylib",
#else
    "liboxiblas_ffi.so", "liboxiblas_ffi.so.0", "liboxiblas_ffi.so.1",
#endif
    NULL
};

/* =========================================================================
 * CBLAS-convention vtable wrappers (CBLAS → Fortran thunks)
 * ======================================================================== */

static void oxiblas_saxpy(int n, float alpha, const float *x, int incx,
                          float *y, int incy)
{
    if (!g_oxiblas.saxpy_f) return;
    float a = alpha;
    g_oxiblas.saxpy_f(&n, &a, x, &incx, y, &incy);
}

static void oxiblas_daxpy(int n, double alpha, const double *x, int incx,
                          double *y, int incy)
{
    if (!g_oxiblas.daxpy_f) return;
    double a = alpha;
    g_oxiblas.daxpy_f(&n, &a, x, &incx, y, &incy);
}

/* CBLAS layout mapping for SGEMM wrapper:
 *   CBLAS order/trans enums → Fortran single-char ('N','T','C')             */
static char trans_char(int t)
{
    if (t == 112 /* CblasTrans */)     return 'T';
    if (t == 113 /* CblasConjTrans */) return 'C';
    return 'N';
}

static void oxiblas_sgemm(int Order, int TransA, int TransB,
                          int M, int N, int K,
                          float alpha, const float *A, int lda,
                          const float *B, int ldb,
                          float beta,  float *C, int ldc)
{
    if (!g_oxiblas.sgemm_f) return;
    /* Fortran expects column-major; if row-major swap A↔B and transpose flags */
    char ta, tb;
    const float *pA, *pB;
    int la, lb, m2, n2;
    if (Order == 101 /* CblasRowMajor */) {
        ta = trans_char(TransB); tb = trans_char(TransA);
        pA = B; la = ldb; pB = A; lb = lda;
        m2 = N; n2 = M;
    } else {
        ta = trans_char(TransA); tb = trans_char(TransB);
        pA = A; la = lda; pB = B; lb = ldb;
        m2 = M; n2 = N;
    }
    g_oxiblas.sgemm_f(&ta, &tb, &m2, &n2, &K,
                      &alpha, pA, &la,
                              pB, &lb,
                      &beta,  C,  &ldc);
}

static void oxiblas_dgemm(int Order, int TransA, int TransB,
                          int M, int N, int K,
                          double alpha, const double *A, int lda,
                          const double *B, int ldb,
                          double beta,  double *C, int ldc)
{
    if (!g_oxiblas.dgemm_f) return;
    char ta, tb;
    const double *pA, *pB;
    int la, lb, m2, n2;
    if (Order == 101) {
        ta = trans_char(TransB); tb = trans_char(TransA);
        pA = B; la = ldb; pB = A; lb = lda;
        m2 = N; n2 = M;
    } else {
        ta = trans_char(TransA); tb = trans_char(TransB);
        pA = A; la = lda; pB = B; lb = ldb;
        m2 = M; n2 = N;
    }
    g_oxiblas.dgemm_f(&ta, &tb, &m2, &n2, &K,
                      &alpha, pA, &la,
                              pB, &lb,
                      &beta,  C,  &ldc);
}

/* =========================================================================
 * Public API
 * ======================================================================== */

bool fb_oxiblas_is_available(void)
{
    for (int i = 0; k_names[i]; ++i) {
        fb_lib_handle_t h = FB_LOAD_LIBRARY(k_names[i]);
        if (!h) continue;
        bool ok = FB_GET_PROC_ADDRESS(h, "saxpy_") != NULL &&
                  FB_GET_PROC_ADDRESS(h, "dgemm_") != NULL;
        FB_FREE_LIBRARY(h);
        if (ok) return true;
    }
    return false;
}

int fb_oxiblas_init(void)
{
    if (g_oxiblas.initialized) return 0;

    fb_lib_handle_t h = NULL;
    for (int i = 0; k_names[i]; ++i) {
        h = FB_LOAD_LIBRARY(k_names[i]);
        if (h) break;
    }
    if (!h) {
        fprintf(stderr, "[OxiBLAS] liboxiblas_ffi not found\n");
        return -1;
    }

    g_oxiblas.saxpy_f = (saxpy_fortran_t) FB_GET_PROC_ADDRESS(h, "saxpy_");
    g_oxiblas.daxpy_f = (daxpy_fortran_t) FB_GET_PROC_ADDRESS(h, "daxpy_");
    g_oxiblas.sgemm_f = (sgemm_fortran_t) FB_GET_PROC_ADDRESS(h, "sgemm_");
    g_oxiblas.dgemm_f = (dgemm_fortran_t) FB_GET_PROC_ADDRESS(h, "dgemm_");

    if (!g_oxiblas.saxpy_f || !g_oxiblas.dgemm_f) {
        fprintf(stderr, "[OxiBLAS] required Fortran symbols missing\n");
        FB_FREE_LIBRARY(h);
        return -2;
    }

    g_oxiblas.handle      = h;
    g_oxiblas.initialized = true;

    /* Populate vtable with CBLAS wrappers */
    memset(&g_oxiblas_vtable, 0, sizeof(g_oxiblas_vtable));
    g_oxiblas_vtable.saxpy  = oxiblas_saxpy;
    g_oxiblas_vtable.daxpy  = oxiblas_daxpy;
    g_oxiblas_vtable.sgemm  = oxiblas_sgemm;
    g_oxiblas_vtable.dgemm  = oxiblas_dgemm;

    return 0;
}

void fb_oxiblas_shutdown(void)
{
    if (!g_oxiblas.initialized) return;
    memset(&g_oxiblas_vtable, 0, sizeof(g_oxiblas_vtable));
    FB_FREE_LIBRARY(g_oxiblas.handle);
    memset(&g_oxiblas, 0, sizeof(g_oxiblas));
}

const fb_backend_vtable_t *fb_oxiblas_get_vtable(void)
{
    if (!g_oxiblas.initialized) return NULL;
    return &g_oxiblas_vtable;
}

const char *fb_oxiblas_get_version(void)
{
    return "0.1.0";   /* OxiBLAS 0.1.0 has no runtime version query symbol */
}
