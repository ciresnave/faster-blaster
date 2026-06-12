/**
 * @file rocblas_backend.c
 * @brief AMD rocBLAS backend — fully dynamic loading, no ROCm headers required.
 *
 * Follows the same pattern as cublas_backend.c:
 *  1. All HIP and rocBLAS types are declared locally (no #include <hip/hip_runtime.h>).
 *  2. Libraries loaded at runtime; absence is handled gracefully (returns NULL vtable).
 *  3. CBLAS-style shims capture a global default handle so the vtable can be
 *     used without any handle argument from the caller.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "rocblas_backend.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#ifdef _WIN32
#  include <windows.h>
#  define RLOAD(name)      ((void*)LoadLibraryA(name))
#  define RPROC(lib, sym)  ((void*)GetProcAddress((HMODULE)(lib), sym))
#else
#  include <dlfcn.h>
#  define RLOAD(name)      dlopen(name, RTLD_LAZY)
#  define RPROC(lib, sym)  dlsym(lib, sym)
#endif

/* ── local type declarations (no ROCm headers needed) ─────────────────────── */

typedef struct _rocblas_handle* rocblas_handle;
typedef struct ihipStream_t*    hipStream_t;

typedef enum {
    rocblas_status_success = 0,
    rocblas_status_invalid_handle,
    rocblas_status_not_implemented,
    rocblas_status_invalid_value,
    rocblas_status_internal_error
} rocblas_status;

typedef enum {
    rocblas_operation_none      = 'N',
    rocblas_operation_transpose = 'T',
    rocblas_operation_conjugate_transpose = 'C'
} rocblas_operation;

typedef enum {
    rocblas_fill_upper = 'U',
    rocblas_fill_lower = 'L'
} rocblas_fill;

typedef enum {
    rocblas_diagonal_non_unit = 'N',
    rocblas_diagonal_unit     = 'U'
} rocblas_diagonal;

typedef enum {
    rocblas_side_left  = 'L',
    rocblas_side_right = 'R'
} rocblas_side;

/* ── rocBLAS lifecycle ────────────────────────────────────────────────────── */
typedef rocblas_status (*rocblas_create_handle_t) (rocblas_handle*);
typedef rocblas_status (*rocblas_destroy_handle_t)(rocblas_handle);
typedef rocblas_status (*rocblas_set_stream_t)    (rocblas_handle, hipStream_t);

/* ── rocBLAS BLAS Level 1 ─────────────────────────────────────────────────── */
typedef rocblas_status (*rocblas_sasum_t) (rocblas_handle, int, const float*,  int, float*);
typedef rocblas_status (*rocblas_dasum_t) (rocblas_handle, int, const double*, int, double*);
typedef rocblas_status (*rocblas_saxpy_t) (rocblas_handle, int, const float*,  const float*,  int, float*,  int);
typedef rocblas_status (*rocblas_daxpy_t) (rocblas_handle, int, const double*, const double*, int, double*, int);
typedef rocblas_status (*rocblas_sdot_t)  (rocblas_handle, int, const float*,  int, const float*,  int, float*);
typedef rocblas_status (*rocblas_ddot_t)  (rocblas_handle, int, const double*, int, const double*, int, double*);
typedef rocblas_status (*rocblas_scopy_t) (rocblas_handle, int, const float*,  int, float*,  int);
typedef rocblas_status (*rocblas_dcopy_t) (rocblas_handle, int, const double*, int, double*, int);
typedef rocblas_status (*rocblas_sscal_t) (rocblas_handle, int, const float*,  float*,  int);
typedef rocblas_status (*rocblas_dscal_t) (rocblas_handle, int, const double*, double*, int);
typedef rocblas_status (*rocblas_snrm2_t) (rocblas_handle, int, const float*,  int, float*);
typedef rocblas_status (*rocblas_dnrm2_t) (rocblas_handle, int, const double*, int, double*);
typedef rocblas_status (*rocblas_sswap_t) (rocblas_handle, int, float*,  int, float*,  int);
typedef rocblas_status (*rocblas_dswap_t) (rocblas_handle, int, double*, int, double*, int);
typedef rocblas_status (*rocblas_isamax_t)(rocblas_handle, int, const float*,  int, int*);
typedef rocblas_status (*rocblas_idamax_t)(rocblas_handle, int, const double*, int, int*);

/* ── rocBLAS BLAS Level 2 ─────────────────────────────────────────────────── */
typedef rocblas_status (*rocblas_sgemv_t)(rocblas_handle, rocblas_operation, int, int,
    const float*, const float*, int, const float*, int, const float*, float*, int);
typedef rocblas_status (*rocblas_dgemv_t)(rocblas_handle, rocblas_operation, int, int,
    const double*, const double*, int, const double*, int, const double*, double*, int);

/* ── rocBLAS BLAS Level 3 ─────────────────────────────────────────────────── */
typedef rocblas_status (*rocblas_sgemm_t)(rocblas_handle, rocblas_operation, rocblas_operation,
    int, int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
typedef rocblas_status (*rocblas_dgemm_t)(rocblas_handle, rocblas_operation, rocblas_operation,
    int, int, int, const double*, const double*, int, const double*, int, const double*, double*, int);
typedef rocblas_status (*rocblas_strsm_t)(rocblas_handle, rocblas_side, rocblas_fill,
    rocblas_operation, rocblas_diagonal, int, int, const float*, const float*, int, float*, int);
typedef rocblas_status (*rocblas_dtrsm_t)(rocblas_handle, rocblas_side, rocblas_fill,
    rocblas_operation, rocblas_diagonal, int, int, const double*, const double*, int, double*, int);
typedef rocblas_status (*rocblas_ssyrk_t)(rocblas_handle, rocblas_fill, rocblas_operation,
    int, int, const float*, const float*, int, const float*, float*, int);
typedef rocblas_status (*rocblas_dsyrk_t)(rocblas_handle, rocblas_fill, rocblas_operation,
    int, int, const double*, const double*, int, const double*, double*, int);
typedef rocblas_status (*rocblas_sgemm_strided_batched_t)(rocblas_handle, rocblas_operation, rocblas_operation,
    int, int, int, const float*, const float*, int, long long, const float*, int, long long,
    const float*, float*, int, long long, int);
typedef rocblas_status (*rocblas_dgemm_strided_batched_t)(rocblas_handle, rocblas_operation, rocblas_operation,
    int, int, int, const double*, const double*, int, long long, const double*, int, long long,
    const double*, double*, int, long long, int);

/* ── HIP runtime ─────────────────────────────────────────────────────────── */
typedef int (*hipGetDeviceCount_t)(int*);
typedef int (*hipSetDevice_t)(int);
typedef int (*hipGetDevice_t)(int*);
typedef int (*hipMalloc_t)(void**, size_t);
typedef int (*hipFree_t)(void*);
typedef int (*hipMemcpy_t)(void*, const void*, size_t, int);
typedef int (*hipMemset_t)(void*, int, size_t);
typedef int (*hipDeviceSynchronize_t)(void);
typedef int (*hipStreamCreate_t)(hipStream_t*);
typedef int (*hipStreamDestroy_t)(hipStream_t);
typedef int (*hipStreamSynchronize_t)(hipStream_t);

/* ── global loader state ─────────────────────────────────────────────────── */
static struct {
    void* hip_handle;
    void* rocblas_handle_lib;

    /* lifecycle */
    rocblas_create_handle_t  create_handle;
    rocblas_destroy_handle_t destroy_handle;
    rocblas_set_stream_t     set_stream;

    /* L1 */
    rocblas_sasum_t  Sasum;
    rocblas_dasum_t  Dasum;
    rocblas_saxpy_t  Saxpy;
    rocblas_daxpy_t  Daxpy;
    rocblas_sdot_t   Sdot;
    rocblas_ddot_t   Ddot;
    rocblas_scopy_t  Scopy;
    rocblas_dcopy_t  Dcopy;
    rocblas_sscal_t  Sscal;
    rocblas_dscal_t  Dscal;
    rocblas_snrm2_t  Snrm2;
    rocblas_dnrm2_t  Dnrm2;
    rocblas_sswap_t  Sswap;
    rocblas_dswap_t  Dswap;
    rocblas_isamax_t Isamax;
    rocblas_idamax_t Idamax;

    /* L2 */
    rocblas_sgemv_t Sgemv;
    rocblas_dgemv_t Dgemv;

    /* L3 */
    rocblas_sgemm_t  Sgemm;
    rocblas_dgemm_t  Dgemm;
    rocblas_strsm_t  Strsm;
    rocblas_dtrsm_t  Dtrsm;
    rocblas_ssyrk_t  Ssyrk;
    rocblas_dsyrk_t  Dsyrk;
    rocblas_sgemm_strided_batched_t Sgemm_strided_batched;
    rocblas_dgemm_strided_batched_t Dgemm_strided_batched;

    /* HIP runtime */
    hipGetDeviceCount_t  GetDeviceCount;
    hipSetDevice_t       SetDevice;
    hipGetDevice_t       GetDevice;
    hipMalloc_t          Malloc;
    hipFree_t            Free;
    hipMemcpy_t          Memcpy;
    hipMemset_t          Memset;
    hipDeviceSynchronize_t DeviceSync;
    hipStreamCreate_t    StreamCreate;
    hipStreamDestroy_t   StreamDestroy;
    hipStreamSynchronize_t StreamSync;

    bool initialized;
} g_rocblas;

static rocblas_handle g_rocblas_default_handle = NULL;

static int rocblas_load_api(void) {
    if (g_rocblas.initialized) return 0;

#ifdef _WIN32
    g_rocblas.hip_handle = RLOAD("amdhip64.dll");
    if (!g_rocblas.hip_handle) g_rocblas.hip_handle = RLOAD("amdhip64_6.dll");
    g_rocblas.rocblas_handle_lib = RLOAD("rocblas64_1.dll");
    if (!g_rocblas.rocblas_handle_lib) g_rocblas.rocblas_handle_lib = RLOAD("rocblas.dll");
#else
    g_rocblas.hip_handle = RLOAD("libamdhip64.so.6");
    if (!g_rocblas.hip_handle) g_rocblas.hip_handle = RLOAD("libamdhip64.so");
    g_rocblas.rocblas_handle_lib = RLOAD("librocblas.so.4");
    if (!g_rocblas.rocblas_handle_lib) g_rocblas.rocblas_handle_lib = RLOAD("librocblas.so");
#endif

    if (!g_rocblas.hip_handle || !g_rocblas.rocblas_handle_lib)
        return -1;

    /* lifecycle */
    g_rocblas.create_handle  = (rocblas_create_handle_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_create_handle");
    g_rocblas.destroy_handle = (rocblas_destroy_handle_t)RPROC(g_rocblas.rocblas_handle_lib, "rocblas_destroy_handle");
    g_rocblas.set_stream     = (rocblas_set_stream_t)    RPROC(g_rocblas.rocblas_handle_lib, "rocblas_set_stream");

    /* L1 */
    g_rocblas.Sasum   = (rocblas_sasum_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_sasum");
    g_rocblas.Dasum   = (rocblas_dasum_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_dasum");
    g_rocblas.Saxpy   = (rocblas_saxpy_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_saxpy");
    g_rocblas.Daxpy   = (rocblas_daxpy_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_daxpy");
    g_rocblas.Sdot    = (rocblas_sdot_t)  RPROC(g_rocblas.rocblas_handle_lib, "rocblas_sdot");
    g_rocblas.Ddot    = (rocblas_ddot_t)  RPROC(g_rocblas.rocblas_handle_lib, "rocblas_ddot");
    g_rocblas.Scopy   = (rocblas_scopy_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_scopy");
    g_rocblas.Dcopy   = (rocblas_dcopy_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_dcopy");
    g_rocblas.Sscal   = (rocblas_sscal_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_sscal");
    g_rocblas.Dscal   = (rocblas_dscal_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_dscal");
    g_rocblas.Snrm2   = (rocblas_snrm2_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_snrm2");
    g_rocblas.Dnrm2   = (rocblas_dnrm2_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_dnrm2");
    g_rocblas.Sswap   = (rocblas_sswap_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_sswap");
    g_rocblas.Dswap   = (rocblas_dswap_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_dswap");
    g_rocblas.Isamax  = (rocblas_isamax_t)RPROC(g_rocblas.rocblas_handle_lib, "rocblas_isamax");
    g_rocblas.Idamax  = (rocblas_idamax_t)RPROC(g_rocblas.rocblas_handle_lib, "rocblas_idamax");

    /* L2 */
    g_rocblas.Sgemv   = (rocblas_sgemv_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_sgemv");
    g_rocblas.Dgemv   = (rocblas_dgemv_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_dgemv");

    /* L3 */
    g_rocblas.Sgemm   = (rocblas_sgemm_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_sgemm");
    g_rocblas.Dgemm   = (rocblas_dgemm_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_dgemm");
    g_rocblas.Strsm   = (rocblas_strsm_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_strsm");
    g_rocblas.Dtrsm   = (rocblas_dtrsm_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_dtrsm");
    g_rocblas.Ssyrk   = (rocblas_ssyrk_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_ssyrk");
    g_rocblas.Dsyrk   = (rocblas_dsyrk_t) RPROC(g_rocblas.rocblas_handle_lib, "rocblas_dsyrk");
    g_rocblas.Sgemm_strided_batched =
        (rocblas_sgemm_strided_batched_t)RPROC(g_rocblas.rocblas_handle_lib, "rocblas_sgemm_strided_batched");
    g_rocblas.Dgemm_strided_batched =
        (rocblas_dgemm_strided_batched_t)RPROC(g_rocblas.rocblas_handle_lib, "rocblas_dgemm_strided_batched");

    /* HIP runtime */
    g_rocblas.GetDeviceCount = (hipGetDeviceCount_t)   RPROC(g_rocblas.hip_handle, "hipGetDeviceCount");
    g_rocblas.SetDevice      = (hipSetDevice_t)         RPROC(g_rocblas.hip_handle, "hipSetDevice");
    g_rocblas.GetDevice      = (hipGetDevice_t)         RPROC(g_rocblas.hip_handle, "hipGetDevice");
    g_rocblas.Malloc         = (hipMalloc_t)            RPROC(g_rocblas.hip_handle, "hipMalloc");
    g_rocblas.Free           = (hipFree_t)              RPROC(g_rocblas.hip_handle, "hipFree");
    g_rocblas.Memcpy         = (hipMemcpy_t)            RPROC(g_rocblas.hip_handle, "hipMemcpy");
    g_rocblas.Memset         = (hipMemset_t)            RPROC(g_rocblas.hip_handle, "hipMemset");
    g_rocblas.DeviceSync     = (hipDeviceSynchronize_t) RPROC(g_rocblas.hip_handle, "hipDeviceSynchronize");
    g_rocblas.StreamCreate   = (hipStreamCreate_t)      RPROC(g_rocblas.hip_handle, "hipStreamCreate");
    g_rocblas.StreamDestroy  = (hipStreamDestroy_t)     RPROC(g_rocblas.hip_handle, "hipStreamDestroy");
    g_rocblas.StreamSync     = (hipStreamSynchronize_t) RPROC(g_rocblas.hip_handle, "hipStreamSynchronize");

    g_rocblas.initialized = true;

    /* Create default handle for device 0 */
    if (g_rocblas.create_handle) {
        if (g_rocblas.create_handle(&g_rocblas_default_handle) != rocblas_status_success) {
            g_rocblas_default_handle = NULL;
        }
    }

    return 0;
}

bool fb_rocblas_is_available(void) {
    return rocblas_load_api() == 0;
}

int fb_rocblas_device_count(void) {
    if (rocblas_load_api() != 0) return 0;
    int count = 0;
    if (g_rocblas.GetDeviceCount) g_rocblas.GetDeviceCount(&count);
    return count;
}

int fb_rocblas_init(int device_id, fb_gpu_context_t* ctx) {
    if (!ctx || rocblas_load_api() != 0) return -1;
    if (device_id < 0) device_id = 0;
    if (g_rocblas.SetDevice) g_rocblas.SetDevice(device_id);

    rocblas_handle* rh = (rocblas_handle*)malloc(sizeof(rocblas_handle));
    if (!rh) return -1;
    if (!g_rocblas.create_handle || g_rocblas.create_handle(rh) != rocblas_status_success) {
        free(rh); return -1;
    }
    ctx->device_id       = device_id;
    ctx->backend_context = rh;
    ctx->default_stream  = NULL;
    ctx->synchronous     = false;
    return 0;
}

void fb_rocblas_shutdown(fb_gpu_context_t* ctx) {
    if (!ctx || !ctx->backend_context) return;
    rocblas_handle* rh = (rocblas_handle*)ctx->backend_context;
    if (g_rocblas.destroy_handle) g_rocblas.destroy_handle(*rh);
    free(rh);
    ctx->backend_context = NULL;
}

/* ── CBLAS-style shims ───────────────────────────────────────────────────── */

static float rocblas_shim_sasum(int n, const float* x, int incx) {
    float r = 0.0f;
    if (g_rocblas.Sasum) g_rocblas.Sasum(g_rocblas_default_handle, n, x, incx, &r);
    return r;
}
static double rocblas_shim_dasum(int n, const double* x, int incx) {
    double r = 0.0;
    if (g_rocblas.Dasum) g_rocblas.Dasum(g_rocblas_default_handle, n, x, incx, &r);
    return r;
}
static void rocblas_shim_saxpy(int n, float a, const float* x, int incx, float* y, int incy) {
    if (g_rocblas.Saxpy) g_rocblas.Saxpy(g_rocblas_default_handle, n, &a, x, incx, y, incy);
}
static void rocblas_shim_daxpy(int n, double a, const double* x, int incx, double* y, int incy) {
    if (g_rocblas.Daxpy) g_rocblas.Daxpy(g_rocblas_default_handle, n, &a, x, incx, y, incy);
}
static float rocblas_shim_sdot(int n, const float* x, int incx, const float* y, int incy) {
    float r = 0.0f;
    if (g_rocblas.Sdot) g_rocblas.Sdot(g_rocblas_default_handle, n, x, incx, y, incy, &r);
    return r;
}
static double rocblas_shim_ddot(int n, const double* x, int incx, const double* y, int incy) {
    double r = 0.0;
    if (g_rocblas.Ddot) g_rocblas.Ddot(g_rocblas_default_handle, n, x, incx, y, incy, &r);
    return r;
}
static void rocblas_shim_scopy(int n, const float* x, int incx, float* y, int incy) {
    if (g_rocblas.Scopy) g_rocblas.Scopy(g_rocblas_default_handle, n, x, incx, y, incy);
}
static void rocblas_shim_dcopy(int n, const double* x, int incx, double* y, int incy) {
    if (g_rocblas.Dcopy) g_rocblas.Dcopy(g_rocblas_default_handle, n, x, incx, y, incy);
}
static void rocblas_shim_sscal(int n, float a, float* x, int incx) {
    if (g_rocblas.Sscal) g_rocblas.Sscal(g_rocblas_default_handle, n, &a, x, incx);
}
static void rocblas_shim_dscal(int n, double a, double* x, int incx) {
    if (g_rocblas.Dscal) g_rocblas.Dscal(g_rocblas_default_handle, n, &a, x, incx);
}
static float rocblas_shim_snrm2(int n, const float* x, int incx) {
    float r = 0.0f;
    if (g_rocblas.Snrm2) g_rocblas.Snrm2(g_rocblas_default_handle, n, x, incx, &r);
    return r;
}
static double rocblas_shim_dnrm2(int n, const double* x, int incx) {
    double r = 0.0;
    if (g_rocblas.Dnrm2) g_rocblas.Dnrm2(g_rocblas_default_handle, n, x, incx, &r);
    return r;
}
static void rocblas_shim_sswap(int n, float* x, int incx, float* y, int incy) {
    if (g_rocblas.Sswap) g_rocblas.Sswap(g_rocblas_default_handle, n, x, incx, y, incy);
}
static void rocblas_shim_dswap(int n, double* x, int incx, double* y, int incy) {
    if (g_rocblas.Dswap) g_rocblas.Dswap(g_rocblas_default_handle, n, x, incx, y, incy);
}
static int rocblas_shim_isamax(int n, const float* x, int incx) {
    int r = 0;
    if (g_rocblas.Isamax) g_rocblas.Isamax(g_rocblas_default_handle, n, x, incx, &r);
    return r - 1; /* rocBLAS is 1-based; CBLAS is 0-based */
}
static int rocblas_shim_idamax(int n, const double* x, int incx) {
    int r = 0;
    if (g_rocblas.Idamax) g_rocblas.Idamax(g_rocblas_default_handle, n, x, incx, &r);
    return r - 1;
}

/* Level 2 */
static void rocblas_shim_sgemv(int order, int transA, int m, int n,
                                float alpha, const float* A, int lda,
                                const float* x, int incx, float beta, float* y, int incy) {
    if (!g_rocblas.Sgemv) return;
    rocblas_operation op = (transA == 111) ? rocblas_operation_none : rocblas_operation_transpose;
    g_rocblas.Sgemv(g_rocblas_default_handle, op, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}
static void rocblas_shim_dgemv(int order, int transA, int m, int n,
                                double alpha, const double* A, int lda,
                                const double* x, int incx, double beta, double* y, int incy) {
    if (!g_rocblas.Dgemv) return;
    rocblas_operation op = (transA == 111) ? rocblas_operation_none : rocblas_operation_transpose;
    g_rocblas.Dgemv(g_rocblas_default_handle, op, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}

/* Level 3 */
static void rocblas_shim_sgemm(int order, int transA, int transB, int m, int n, int k,
                                float alpha, const float* A, int lda,
                                const float* B, int ldb, float beta, float* C, int ldc) {
    if (!g_rocblas.Sgemm) return;
    rocblas_operation opA = (transA == 111) ? rocblas_operation_none : rocblas_operation_transpose;
    rocblas_operation opB = (transB == 111) ? rocblas_operation_none : rocblas_operation_transpose;
    if (order == 101) { /* CblasRowMajor -> column-major: swap A/B and m/n */
        g_rocblas.Sgemm(g_rocblas_default_handle, opB, opA, n, m, k,
                        &alpha, B, ldb, A, lda, &beta, C, ldc);
    } else {
        g_rocblas.Sgemm(g_rocblas_default_handle, opA, opB, m, n, k,
                        &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
}
static void rocblas_shim_dgemm(int order, int transA, int transB, int m, int n, int k,
                                double alpha, const double* A, int lda,
                                const double* B, int ldb, double beta, double* C, int ldc) {
    if (!g_rocblas.Dgemm) return;
    rocblas_operation opA = (transA == 111) ? rocblas_operation_none : rocblas_operation_transpose;
    rocblas_operation opB = (transB == 111) ? rocblas_operation_none : rocblas_operation_transpose;
    if (order == 101) {
        g_rocblas.Dgemm(g_rocblas_default_handle, opB, opA, n, m, k,
                        &alpha, B, ldb, A, lda, &beta, C, ldc);
    } else {
        g_rocblas.Dgemm(g_rocblas_default_handle, opA, opB, m, n, k,
                        &alpha, A, lda, B, ldb, &beta, C, ldc);
    }
}
static void rocblas_shim_strsm(int order, int side, int uplo, int transA, int diag,
                                int m, int n, float alpha, const float* A, int lda, float* B, int ldb) {
    if (!g_rocblas.Strsm) return;
    rocblas_side      s  = (side == 141) ? rocblas_side_left  : rocblas_side_right;
    rocblas_fill      u  = (uplo == 121) ? rocblas_fill_upper : rocblas_fill_lower;
    rocblas_operation op = (transA == 111) ? rocblas_operation_none : rocblas_operation_transpose;
    rocblas_diagonal  dt = (diag == 131)  ? rocblas_diagonal_non_unit : rocblas_diagonal_unit;
    g_rocblas.Strsm(g_rocblas_default_handle, s, u, op, dt, m, n, &alpha, A, lda, B, ldb);
}
static void rocblas_shim_dtrsm(int order, int side, int uplo, int transA, int diag,
                                int m, int n, double alpha, const double* A, int lda, double* B, int ldb) {
    if (!g_rocblas.Dtrsm) return;
    rocblas_side      s  = (side == 141) ? rocblas_side_left  : rocblas_side_right;
    rocblas_fill      u  = (uplo == 121) ? rocblas_fill_upper : rocblas_fill_lower;
    rocblas_operation op = (transA == 111) ? rocblas_operation_none : rocblas_operation_transpose;
    rocblas_diagonal  dt = (diag == 131)  ? rocblas_diagonal_non_unit : rocblas_diagonal_unit;
    g_rocblas.Dtrsm(g_rocblas_default_handle, s, u, op, dt, m, n, &alpha, A, lda, B, ldb);
}
static void rocblas_shim_ssyrk(int order, int uplo, int trans, int n, int k,
                                float alpha, const float* A, int lda, float beta, float* C, int ldc) {
    if (!g_rocblas.Ssyrk) return;
    rocblas_fill u = (uplo == 121) ? rocblas_fill_upper : rocblas_fill_lower;
    rocblas_operation op = (trans == 111) ? rocblas_operation_none : rocblas_operation_transpose;
    g_rocblas.Ssyrk(g_rocblas_default_handle, u, op, n, k, &alpha, A, lda, &beta, C, ldc);
}
static void rocblas_shim_dsyrk(int order, int uplo, int trans, int n, int k,
                                double alpha, const double* A, int lda, double beta, double* C, int ldc) {
    if (!g_rocblas.Dsyrk) return;
    rocblas_fill u = (uplo == 121) ? rocblas_fill_upper : rocblas_fill_lower;
    rocblas_operation op = (trans == 111) ? rocblas_operation_none : rocblas_operation_transpose;
    g_rocblas.Dsyrk(g_rocblas_default_handle, u, op, n, k, &alpha, A, lda, &beta, C, ldc);
}

/* Memory management */
static int  rb_mem_alloc(void* bh, void** ptr, size_t sz) {
    (void)bh; return (g_rocblas.Malloc && g_rocblas.Malloc(ptr, sz) == 0) ? 0 : -1;
}
static void rb_mem_free(void* bh, void* ptr) {
    (void)bh; if (g_rocblas.Free) g_rocblas.Free(ptr);
}
static int rb_mem_upload(void* bh, void* dst, const void* src, size_t sz) {
    (void)bh; return g_rocblas.Memcpy ? g_rocblas.Memcpy(dst, src, sz, 1) : -1; /* H2D=1 */
}
static int rb_mem_download(void* bh, void* dst, const void* src, size_t sz) {
    (void)bh; return g_rocblas.Memcpy ? g_rocblas.Memcpy(dst, src, sz, 2) : -1; /* D2H=2 */
}
static int rb_mem_copy_dev(void* bh, void* dst, const void* src, size_t sz) {
    (void)bh; return g_rocblas.Memcpy ? g_rocblas.Memcpy(dst, src, sz, 3) : -1; /* D2D=3 */
}
static int  rb_stream_create(void* bh, void** s) {
    (void)bh; return g_rocblas.StreamCreate ? g_rocblas.StreamCreate((hipStream_t*)s) : -1;
}
static void rb_stream_destroy(void* bh, void* s) {
    (void)bh; if (g_rocblas.StreamDestroy) g_rocblas.StreamDestroy((hipStream_t)s);
}
static int  rb_stream_sync(void* bh, void* s) {
    (void)bh; return g_rocblas.StreamSync ? g_rocblas.StreamSync((hipStream_t)s) : 0;
}
static int  rb_stream_set(void* bh, void* s) {
    (void)bh;
    return (g_rocblas_default_handle && g_rocblas.set_stream)
               ? (int)g_rocblas.set_stream(g_rocblas_default_handle, (hipStream_t)s) : -1;
}
static uint32_t rb_get_capabilities(void* bh) {
    (void)bh;
    return FB_CAP_GPU | FB_CAP_LEVEL1 | FB_CAP_LEVEL2 | FB_CAP_LEVEL3
         | FB_CAP_SINGLE | FB_CAP_DOUBLE;
}

/* ── vtable ──────────────────────────────────────────────────────────────── */
static fb_backend_vtable_t g_rocblas_vtable = {
    .sasum  = rocblas_shim_sasum,
    .dasum  = rocblas_shim_dasum,
    .saxpy  = rocblas_shim_saxpy,
    .daxpy  = rocblas_shim_daxpy,
    .sdot   = rocblas_shim_sdot,
    .ddot   = rocblas_shim_ddot,
    .scopy  = rocblas_shim_scopy,
    .dcopy  = rocblas_shim_dcopy,
    .sscal  = rocblas_shim_sscal,
    .dscal  = rocblas_shim_dscal,
    .snrm2  = rocblas_shim_snrm2,
    .dnrm2  = rocblas_shim_dnrm2,
    .sswap  = rocblas_shim_sswap,
    .dswap  = rocblas_shim_dswap,
    .isamax = rocblas_shim_isamax,
    .idamax = rocblas_shim_idamax,
    .sgemv  = rocblas_shim_sgemv,
    .dgemv  = rocblas_shim_dgemv,
    .sgemm  = rocblas_shim_sgemm,
    .dgemm  = rocblas_shim_dgemm,
    .strsm  = rocblas_shim_strsm,
    .dtrsm  = rocblas_shim_dtrsm,
    .ssyrk  = rocblas_shim_ssyrk,
    .dsyrk  = rocblas_shim_dsyrk,
    .mem_alloc    = rb_mem_alloc,
    .mem_free     = rb_mem_free,
    .mem_upload   = rb_mem_upload,
    .mem_download = rb_mem_download,
    .mem_copy     = rb_mem_copy_dev,
    .stream_create  = rb_stream_create,
    .stream_destroy = rb_stream_destroy,
    .stream_sync    = rb_stream_sync,
    .stream_set     = rb_stream_set,
    .get_capabilities = rb_get_capabilities,
    .get_num_threads  = NULL,
    .set_num_threads  = NULL,
};

const fb_gpu_backend_trait_t* fb_rocblas_get_backend(void) {
    if (!fb_rocblas_is_available()) return NULL;
    return NULL; /* fb_gpu_backend_trait_t layer not yet wired */
}

const fb_backend_vtable_t* fb_rocblas_get_vtable(void) {
    if (!fb_rocblas_is_available() || !g_rocblas_default_handle) return NULL;
    /* ext_ops[]: same constraint as cuBLAS — rocBLAS functions take a handle
     * as first arg, incompatible with CBLAS-style ext_ops dispatch.  Extended
     * ops (rocSOLVER etc.) will be added via dedicated shims when needed. */
    return &g_rocblas_vtable;
}

int fb_rocblas_get_device_info(int device_id, fb_gpu_device_info_t* info) {
    (void)device_id; (void)info;
    return -1; /* TODO: hipDeviceGetAttribute */
}
