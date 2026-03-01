/**
 * @file conv_thunks.c
 * @brief Cross-convention calling-convention thunks for BLAS/LAPACK operations.
 *
 * ## Design
 *
 * One static thunk function is generated per (operation, direction) pair for
 * every standard BLAS operation that differs between calling conventions.
 *
 * Two arrays are exposed:
 *   - k_cblas_to_fortran_thunks[op]   — wraps CBLAS impl, exposes Fortran ABI
 *   - k_fortran_to_cblas_thunks[op]   — wraps Fortran impl, exposes CBLAS ABI
 *
 * Backing function pointers (g_cblas_fn / g_fortran_fn) are filled by
 * fb_install_conv_thunks() which is called from vtable_autofill.c Strategy 5.
 *
 * ## Key difference: CBLAS vs Fortran
 *
 *   CBLAS:   pass-by-value for integer and real scalar IN arguments
 *   Fortran: pass-by-pointer for ALL arguments (integers, scalars, arrays)
 *
 * Complex scalars are already passed by pointer in CBLAS (as const void *), so
 * complex-typed operations only need to convert the integer parameters.
 *
 * ## Note on multi-backend support
 *
 * The global backing arrays (g_cblas_fn, g_fortran_fn) hold one function
 * pointer per operation, limiting full thunk support to one active backend at
 * a time.  Multi-backend JIT trampolines are a planned enhancement.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#include "conv_thunks.h"

#include <stddef.h>
#include <string.h>
#include "../backends/backend_interface.h"
#include "../judge/judge_op_ids.h"

/* ============================================================================
 * Global backing pointers
 *
 * g_cblas_fn[op]    — CBLAS-convention implementation (used by →Fortran thunks)
 * g_fortran_fn[op]  — Fortran-convention implementation (used by →CBLAS thunks)
 * ========================================================================== */
static fb_generic_fn g_cblas_fn   [FB_JUDGE_MAX_OPERATIONS];
static fb_generic_fn g_fortran_fn [FB_JUDGE_MAX_OPERATIONS];

/* ============================================================================
 * Thunk helper macros
 *
 * These generate a pair of static thunk functions for operations that follow
 * the common integer + real-scalar pattern:
 *   cblas_OP (N, ALPHA, X, INCX, Y, INCY)
 *   OP_      (N*, ALPHA*, X, INCX*, Y, INCY*)
 * ========================================================================== */

/** AXPY family: (n, alpha:scalar, x, incx, y, incy) */
#define DEFINE_AXPY_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int n, TYPE alpha, const TYPE *x, int incx, TYPE *y, int incy) {            \
    int  n_ = n, incx_ = incx, incy_ = incy;                                   \
    TYPE a_ = alpha;                                                             \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    ((void(*)(const int*, const TYPE*, const TYPE*,                             \
              const int*, TYPE*, const int*))                                    \
     g_fortran_fn[OP_ID])(&n_, &a_, x, &incx_, y, &incy_);                     \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *n, const TYPE *alpha, const TYPE *x,                             \
    const int *incx, TYPE *y, const int *incy) {                                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, TYPE, const TYPE*, int, TYPE*, int))                         \
     g_cblas_fn[OP_ID])(*n, *alpha, x, *incx, y, *incy);                       \
}

/** SCAL family: (n, alpha:scalar, x, incx) */
#define DEFINE_SCAL_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int n, TYPE alpha, TYPE *x, int incx) {                                     \
    int  n_ = n, incx_ = incx;                                                  \
    TYPE a_ = alpha;                                                             \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    ((void(*)(const int*, const TYPE*, TYPE*, const int*))                      \
     g_fortran_fn[OP_ID])(&n_, &a_, x, &incx_);                                \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *n, const TYPE *alpha, TYPE *x, const int *incx) {                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, TYPE, TYPE*, int))                                            \
     g_cblas_fn[OP_ID])(*n, *alpha, x, *incx);                                 \
}

/** NRM2/ASUM family: (n, x, incx) → scalar RETURN (same register ABI) */
#define DEFINE_NRM2_THUNKS(OP, RTYPE, TYPE, OP_ID)                             \
static RTYPE thunk_##OP##_cblas_to_fortran(                                     \
    int n, const TYPE *x, int incx) {                                           \
    int n_ = n, incx_ = incx;                                                   \
    if (!g_fortran_fn[OP_ID]) return (RTYPE)0;                                  \
    return ((RTYPE(*)(const int*, const TYPE*, const int*))                     \
     g_fortran_fn[OP_ID])(&n_, x, &incx_);                                     \
}                                                                               \
static RTYPE thunk_##OP##_fortran_to_cblas(                                     \
    const int *n, const TYPE *x, const int *incx) {                             \
    if (!g_cblas_fn[OP_ID]) return (RTYPE)0;                                    \
    return ((RTYPE(*)(int, const TYPE*, int))                                   \
     g_cblas_fn[OP_ID])(*n, x, *incx);                                         \
}

/** DOT family: (n, x, incx, y, incy) → scalar RETURN */
#define DEFINE_DOT_THUNKS(OP, TYPE, OP_ID)                                     \
static TYPE thunk_##OP##_cblas_to_fortran(                                      \
    int n, const TYPE *x, int incx, const TYPE *y, int incy) {                  \
    int n_ = n, incx_ = incx, incy_ = incy;                                    \
    if (!g_fortran_fn[OP_ID]) return (TYPE)0;                                   \
    return ((TYPE(*)(const int*, const TYPE*, const int*,                       \
                     const TYPE*, const int*))                                   \
     g_fortran_fn[OP_ID])(&n_, x, &incx_, y, &incy_);                          \
}                                                                               \
static TYPE thunk_##OP##_fortran_to_cblas(                                      \
    const int *n, const TYPE *x, const int *incx,                               \
    const TYPE *y, const int *incy) {                                            \
    if (!g_cblas_fn[OP_ID]) return (TYPE)0;                                     \
    return ((TYPE(*)(int, const TYPE*, int, const TYPE*, int))                  \
     g_cblas_fn[OP_ID])(*n, x, *incx, y, *incy);                               \
}

/** COPY/SWAP family: (n, x, incx, y, incy) — no scalar */
#define DEFINE_COPY_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int n, const TYPE *x, int incx, TYPE *y, int incy) {                        \
    int n_ = n, incx_ = incx, incy_ = incy;                                    \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    ((void(*)(const int*, const TYPE*, const int*, TYPE*, const int*))          \
     g_fortran_fn[OP_ID])(&n_, x, &incx_, y, &incy_);                          \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *n, const TYPE *x, const int *incx, TYPE *y, const int *incy) {  \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, const TYPE*, int, TYPE*, int))                               \
     g_cblas_fn[OP_ID])(*n, x, *incx, y, *incy);                               \
}

/* Integer-only pointer conversion for complex AXPY (alpha is const void *,
 * same in CBLAS and Fortran — only n/incx/incy need pointer wrapping). */
#define DEFINE_CAXPY_THUNKS(OP, OP_ID)                                         \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int n, const void *alpha, const void *x, int incx, void *y, int incy) {    \
    int n_ = n, incx_ = incx, incy_ = incy;                                    \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    ((void(*)(const int*, const void*, const void*,                             \
              const int*, void*, const int*))                                    \
     g_fortran_fn[OP_ID])(&n_, alpha, x, &incx_, y, &incy_);                   \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *n, const void *alpha, const void *x,                             \
    const int *incx, void *y, const int *incy) {                                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, const void*, const void*, int, void*, int))                  \
     g_cblas_fn[OP_ID])(*n, alpha, x, *incx, y, *incy);                        \
}

/* ============================================================================
 * Thunk instantiation — BLAS Level 1
 * ========================================================================== */

DEFINE_AXPY_THUNKS(saxpy, float,  FB_OP_SAXPY)
DEFINE_AXPY_THUNKS(daxpy, double, FB_OP_DAXPY)
DEFINE_CAXPY_THUNKS(caxpy, FB_OP_CAXPY)
DEFINE_CAXPY_THUNKS(zaxpy, FB_OP_ZAXPY)

DEFINE_SCAL_THUNKS(sscal, float,  FB_OP_SSCAL)
DEFINE_SCAL_THUNKS(dscal, double, FB_OP_DSCAL)

DEFINE_COPY_THUNKS(scopy, float,  FB_OP_SCOPY)
DEFINE_COPY_THUNKS(dcopy, double, FB_OP_DCOPY)
DEFINE_COPY_THUNKS(sswap, float,  FB_OP_SSWAP)
DEFINE_COPY_THUNKS(dswap, double, FB_OP_DSWAP)

DEFINE_NRM2_THUNKS(snrm2,  float,  float,  FB_OP_SNRM2)
DEFINE_NRM2_THUNKS(dnrm2,  double, double, FB_OP_DNRM2)
DEFINE_NRM2_THUNKS(scnrm2, float,  float,  FB_OP_SCNRM2)
DEFINE_NRM2_THUNKS(dznrm2, double, double, FB_OP_DZNRM2)
DEFINE_NRM2_THUNKS(sasum,  float,  float,  FB_OP_SASUM)
DEFINE_NRM2_THUNKS(dasum,  double, double, FB_OP_DASUM)
DEFINE_NRM2_THUNKS(scasum, float,  float,  FB_OP_SCASUM)
DEFINE_NRM2_THUNKS(dzasum, double, double, FB_OP_DZASUM)

DEFINE_DOT_THUNKS(sdot, float,  FB_OP_SDOT)
DEFINE_DOT_THUNKS(ddot, double, FB_OP_DDOT)

/* ============================================================================
 * BLAS Level 3 — SGEMM / DGEMM
 *
 * Full signature (CBLAS order_/transA/transB enums are int-compatible):
 *   sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
 * Fortran convention wraps all scalar + integer args as pointers.
 * ========================================================================== */

static void thunk_sgemm_cblas_to_fortran(
    int Order, int TransA, int TransB,
    int M, int N, int K,
    float alpha, const float *A, int lda,
                 const float *B, int ldb,
    float beta, float *C, int ldc)
{
    if (!g_fortran_fn[FB_OP_SGEMM]) return;
    int Order_  = Order,  TransA_ = TransA, TransB_ = TransB;
    int M_ = M, N_ = N, K_ = K;
    float alpha_ = alpha, beta_ = beta;
    int lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    ((void(*)(const int*, const int*, const int*,
              const int*, const int*, const int*,
              const float*, const float*, const int*,
                            const float*, const int*,
              const float*, float*, const int*))
     g_fortran_fn[FB_OP_SGEMM])(
         &Order_, &TransA_, &TransB_,
         &M_, &N_, &K_,
         &alpha_, A, &lda_, B, &ldb_, &beta_, C, &ldc_);
}

static void thunk_sgemm_fortran_to_cblas(
    const int *Order, const int *TransA, const int *TransB,
    const int *M,     const int *N,      const int *K,
    const float *alpha, const float *A, const int *lda,
                        const float *B, const int *ldb,
    const float *beta, float *C, const int *ldc)
{
    if (!g_cblas_fn[FB_OP_SGEMM]) return;
    ((void(*)(int, int, int, int, int, int,
              float, const float*, int, const float*, int,
              float, float*, int))
     g_cblas_fn[FB_OP_SGEMM])(
         *Order, *TransA, *TransB,
         *M, *N, *K,
         *alpha, A, *lda, B, *ldb, *beta, C, *ldc);
}

static void thunk_dgemm_cblas_to_fortran(
    int Order, int TransA, int TransB,
    int M, int N, int K,
    double alpha, const double *A, int lda,
                  const double *B, int ldb,
    double beta, double *C, int ldc)
{
    if (!g_fortran_fn[FB_OP_DGEMM]) return;
    int Order_  = Order,  TransA_ = TransA, TransB_ = TransB;
    int M_ = M, N_ = N, K_ = K;
    double alpha_ = alpha, beta_ = beta;
    int lda_ = lda, ldb_ = ldb, ldc_ = ldc;
    ((void(*)(const int*, const int*, const int*,
              const int*, const int*, const int*,
              const double*, const double*, const int*,
                             const double*, const int*,
              const double*, double*, const int*))
     g_fortran_fn[FB_OP_DGEMM])(
         &Order_, &TransA_, &TransB_,
         &M_, &N_, &K_,
         &alpha_, A, &lda_, B, &ldb_, &beta_, C, &ldc_);
}

static void thunk_dgemm_fortran_to_cblas(
    const int *Order, const int *TransA, const int *TransB,
    const int *M,     const int *N,      const int *K,
    const double *alpha, const double *A, const int *lda,
                         const double *B, const int *ldb,
    const double *beta, double *C, const int *ldc)
{
    if (!g_cblas_fn[FB_OP_DGEMM]) return;
    ((void(*)(int, int, int, int, int, int,
              double, const double*, int, const double*, int,
              double, double*, int))
     g_cblas_fn[FB_OP_DGEMM])(
         *Order, *TransA, *TransB,
         *M, *N, *K,
         *alpha, A, *lda, B, *ldb, *beta, C, *ldc);
}

/* ============================================================================
 * Thunk dispatch tables
 *
 * All positions default to NULL; the explicit entries are filled by the
 * designated initialisers below.  Positions without a thunk are skipped by
 * fb_install_conv_thunks() (NULL check).
 * ========================================================================== */

/* Clang/GCC designated initialisers allow sparse initialisation of the array.
 * On MSVC use __declspec(selectany) + explicit index assignments instead.    */
#if defined(_MSC_VER)
/* MSVC: initialise to all-zero, then patch in fb_install_conv_thunks. */
fb_generic_fn const k_cblas_to_fortran_thunks[FB_JUDGE_MAX_OPERATIONS];
fb_generic_fn const k_fortran_to_cblas_thunks[FB_JUDGE_MAX_OPERATIONS];

/* Flag to defer MSVC array setup to runtime init (see fb_install_conv_thunks). */
#define FB_THUNK_TABLES_MSVC 1
#else
/* GCC / Clang — use designated initialisers for a sparse const array. */
#  define C2F(fn) ((fb_generic_fn)(void(*)(void))(fn))
#  define F2C(fn) ((fb_generic_fn)(void(*)(void))(fn))

fb_generic_fn const k_cblas_to_fortran_thunks[FB_JUDGE_MAX_OPERATIONS] = {
    [FB_OP_SAXPY]  = C2F(thunk_saxpy_cblas_to_fortran),
    [FB_OP_DAXPY]  = C2F(thunk_daxpy_cblas_to_fortran),
    [FB_OP_CAXPY]  = C2F(thunk_caxpy_cblas_to_fortran),
    [FB_OP_ZAXPY]  = C2F(thunk_zaxpy_cblas_to_fortran),
    [FB_OP_SSCAL]  = C2F(thunk_sscal_cblas_to_fortran),
    [FB_OP_DSCAL]  = C2F(thunk_dscal_cblas_to_fortran),
    [FB_OP_SCOPY]  = C2F(thunk_scopy_cblas_to_fortran),
    [FB_OP_DCOPY]  = C2F(thunk_dcopy_cblas_to_fortran),
    [FB_OP_SSWAP]  = C2F(thunk_sswap_cblas_to_fortran),
    [FB_OP_DSWAP]  = C2F(thunk_dswap_cblas_to_fortran),
    [FB_OP_SNRM2]  = C2F(thunk_snrm2_cblas_to_fortran),
    [FB_OP_DNRM2]  = C2F(thunk_dnrm2_cblas_to_fortran),
    [FB_OP_SCNRM2] = C2F(thunk_scnrm2_cblas_to_fortran),
    [FB_OP_DZNRM2] = C2F(thunk_dznrm2_cblas_to_fortran),
    [FB_OP_SASUM]  = C2F(thunk_sasum_cblas_to_fortran),
    [FB_OP_DASUM]  = C2F(thunk_dasum_cblas_to_fortran),
    [FB_OP_SCASUM] = C2F(thunk_scasum_cblas_to_fortran),
    [FB_OP_DZASUM] = C2F(thunk_dzasum_cblas_to_fortran),
    [FB_OP_SDOT]   = C2F(thunk_sdot_cblas_to_fortran),
    [FB_OP_DDOT]   = C2F(thunk_ddot_cblas_to_fortran),
    [FB_OP_SGEMM]  = C2F(thunk_sgemm_cblas_to_fortran),
    [FB_OP_DGEMM]  = C2F(thunk_dgemm_cblas_to_fortran),
};

fb_generic_fn const k_fortran_to_cblas_thunks[FB_JUDGE_MAX_OPERATIONS] = {
    [FB_OP_SAXPY]  = F2C(thunk_saxpy_fortran_to_cblas),
    [FB_OP_DAXPY]  = F2C(thunk_daxpy_fortran_to_cblas),
    [FB_OP_CAXPY]  = F2C(thunk_caxpy_fortran_to_cblas),
    [FB_OP_ZAXPY]  = F2C(thunk_zaxpy_fortran_to_cblas),
    [FB_OP_SSCAL]  = F2C(thunk_sscal_fortran_to_cblas),
    [FB_OP_DSCAL]  = F2C(thunk_dscal_fortran_to_cblas),
    [FB_OP_SCOPY]  = F2C(thunk_scopy_fortran_to_cblas),
    [FB_OP_DCOPY]  = F2C(thunk_dcopy_fortran_to_cblas),
    [FB_OP_SSWAP]  = F2C(thunk_sswap_fortran_to_cblas),
    [FB_OP_DSWAP]  = F2C(thunk_dswap_fortran_to_cblas),
    [FB_OP_SNRM2]  = F2C(thunk_snrm2_fortran_to_cblas),
    [FB_OP_DNRM2]  = F2C(thunk_dnrm2_fortran_to_cblas),
    [FB_OP_SCNRM2] = F2C(thunk_scnrm2_fortran_to_cblas),
    [FB_OP_DZNRM2] = F2C(thunk_dznrm2_fortran_to_cblas),
    [FB_OP_SASUM]  = F2C(thunk_sasum_fortran_to_cblas),
    [FB_OP_DASUM]  = F2C(thunk_dasum_fortran_to_cblas),
    [FB_OP_SCASUM] = F2C(thunk_scasum_fortran_to_cblas),
    [FB_OP_DZASUM] = F2C(thunk_dzasum_fortran_to_cblas),
    [FB_OP_SDOT]   = F2C(thunk_sdot_fortran_to_cblas),
    [FB_OP_DDOT]   = F2C(thunk_ddot_fortran_to_cblas),
    [FB_OP_SGEMM]  = F2C(thunk_sgemm_fortran_to_cblas),
    [FB_OP_DGEMM]  = F2C(thunk_dgemm_fortran_to_cblas),
};

#  undef C2F
#  undef F2C
#endif /* !_MSC_VER */

/* ============================================================================
 * fb_install_conv_thunks
 * ========================================================================== */

void fb_install_conv_thunks(fb_backend_vtable_t *vtable, uint32_t op_id)
{
    if (!vtable || op_id >= (uint32_t)FB_JUDGE_MAX_OPERATIONS) return;

    fb_generic_fn cblas_fn   = vtable->ext_ops[op_id][FB_CONV_CBLAS];
    fb_generic_fn fortran_fn = vtable->ext_ops[op_id][FB_CONV_FORTRAN];

    /* Record which implementations are available for thunks to delegate to. */
    if (cblas_fn   != NULL) g_cblas_fn[op_id]   = cblas_fn;
    if (fortran_fn != NULL) g_fortran_fn[op_id]  = fortran_fn;

    /* Fill CBLAS → Fortran if CBLAS is present but Fortran is not. */
    if (cblas_fn != NULL && fortran_fn == NULL) {
        fb_generic_fn thunk = k_cblas_to_fortran_thunks[op_id];
        if (thunk != NULL) {
            vtable->ext_ops[op_id][FB_CONV_FORTRAN] = thunk;
        }
    }

    /* Fill Fortran → CBLAS if Fortran is present but CBLAS is not. */
    if (fortran_fn != NULL && cblas_fn == NULL) {
        fb_generic_fn thunk = k_fortran_to_cblas_thunks[op_id];
        if (thunk != NULL) {
            vtable->ext_ops[op_id][FB_CONV_CBLAS] = thunk;
        }
    }

    /* REF shares the CBLAS ABI — direct alias, no thunk needed. */
    if (vtable->ext_ops[op_id][FB_CONV_REF]  != NULL &&
        vtable->ext_ops[op_id][FB_CONV_CBLAS] == NULL) {
        vtable->ext_ops[op_id][FB_CONV_CBLAS] =
            vtable->ext_ops[op_id][FB_CONV_REF];
    }
    if (vtable->ext_ops[op_id][FB_CONV_CBLAS] != NULL &&
        vtable->ext_ops[op_id][FB_CONV_REF]   == NULL) {
        vtable->ext_ops[op_id][FB_CONV_REF] =
            vtable->ext_ops[op_id][FB_CONV_CBLAS];
    }
}
