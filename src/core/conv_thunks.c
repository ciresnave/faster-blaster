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


/** ROT family: (n, x, incx, y, incy, c, s) */
#define DEFINE_ROT_THUNKS(OP, TYPE, OP_ID)                                      \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int n, TYPE *x, int incx, TYPE *y, int incy, TYPE c, TYPE s) {              \
    int n_ = n, incx_ = incx, incy_ = incy;                                    \
    TYPE c_ = c, s_ = s;                                                        \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    ((void(*)(const int*, TYPE*, const int*, TYPE*, const int*,                 \
              const TYPE*, const TYPE*))                                         \
     g_fortran_fn[OP_ID])(&n_, x, &incx_, y, &incy_, &c_, &s_);               \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *n, TYPE *x, const int *incx, TYPE *y, const int *incy,          \
    const TYPE *c, const TYPE *s) {                                             \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, TYPE*, int, TYPE*, int, TYPE, TYPE))                         \
     g_cblas_fn[OP_ID])(*n, x, *incx, y, *incy, *c, *s);                      \
}

/** IAMAX family: (n, x, incx) → int */
#define DEFINE_IAMAX_THUNKS(OP, TYPE, OP_ID)                                   \
static int thunk_##OP##_cblas_to_fortran(                                       \
    int n, const TYPE *x, int incx) {                                           \
    int n_ = n, incx_ = incx;                                                   \
    if (!g_fortran_fn[OP_ID]) return 0;                                         \
    return ((int(*)(const int*, const TYPE*, const int*))                       \
     g_fortran_fn[OP_ID])(&n_, x, &incx_);                                     \
}                                                                               \
static int thunk_##OP##_fortran_to_cblas(                                       \
    const int *n, const TYPE *x, const int *incx) {                             \
    if (!g_cblas_fn[OP_ID]) return 0;                                           \
    return ((int(*)(int, const TYPE*, int))                                     \
     g_cblas_fn[OP_ID])(*n, x, *incx);                                         \
}

/* ============================================================================
 * Thunk helper macros — BLAS Level 2
 * ========================================================================== */

/** GEMV: (Order, Trans, M, N, alpha, A, lda, X, incx, beta, Y, incy) */
#define DEFINE_GEMV_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Trans, int M, int N, TYPE alpha,                             \
    const TYPE *A, int lda, const TYPE *X, int incx,                            \
    TYPE beta, TYPE *Y, int incy) {                                             \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Trans_=Trans, M_=M, N_=N;                                \
    TYPE alpha_=alpha, beta_=beta;                                              \
    int lda_=lda, incx_=incx, incy_=incy;                                      \
    ((void(*)(const int*, const int*, const int*, const int*,                   \
              const TYPE*, const TYPE*, const int*,                              \
              const TYPE*, const int*, const TYPE*, TYPE*, const int*))          \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Trans_, &M_, &N_,                                            \
         &alpha_, A, &lda_, X, &incx_, &beta_, Y, &incy_);                     \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Trans, const int *M, const int *N,             \
    const TYPE *alpha, const TYPE *A, const int *lda,                           \
    const TYPE *X, const int *incx, const TYPE *beta, TYPE *Y,                  \
    const int *incy) {                                                          \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, TYPE, const TYPE*, int,                       \
              const TYPE*, int, TYPE, TYPE*, int))                               \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *Trans, *M, *N,                                                \
         *alpha, A, *lda, X, *incx, *beta, Y, *incy);                          \
}

/** GER: (Order, M, N, alpha, X, incx, Y, incy, A, lda) */
#define DEFINE_GER_THUNKS(OP, TYPE, OP_ID)                                     \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int M, int N, TYPE alpha,                                        \
    const TYPE *X, int incx, const TYPE *Y, int incy, TYPE *A, int lda) {       \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, M_=M, N_=N, incx_=incx, incy_=incy, lda_=lda;           \
    TYPE alpha_=alpha;                                                          \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
              const TYPE*, const int*, const TYPE*, const int*,                 \
              TYPE*, const int*))                                                \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &M_, &N_, &alpha_,                                            \
         X, &incx_, Y, &incy_, A, &lda_);                                      \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *M, const int *N, const TYPE *alpha,            \
    const TYPE *X, const int *incx, const TYPE *Y, const int *incy,             \
    TYPE *A, const int *lda) {                                                  \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, int, const TYPE*, int,          \
              TYPE*, int))                                                       \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *M, *N, *alpha, X, *incx, Y, *incy, A, *lda);                 \
}

/** SYR: (Order, Uplo, N, alpha, X, incx, A, lda) */
#define DEFINE_SYR_THUNKS(OP, TYPE, OP_ID)                                     \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Uplo, int N, TYPE alpha,                                     \
    const TYPE *X, int incx, TYPE *A, int lda) {                                \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Uplo_=Uplo, N_=N, incx_=incx, lda_=lda;                 \
    TYPE alpha_=alpha;                                                          \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
              const TYPE*, const int*, TYPE*, const int*))                       \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &N_, &alpha_, X, &incx_, A, &lda_);                  \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Uplo, const int *N, const TYPE *alpha,         \
    const TYPE *X, const int *incx, TYPE *A, const int *lda) {                  \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, int, TYPE*, int))               \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, *alpha, X, *incx, A, *lda);         \
}

/** SYR2: (Order, Uplo, N, alpha, X, incx, Y, incy, A, lda) */
#define DEFINE_SYR2_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Uplo, int N, TYPE alpha,                                     \
    const TYPE *X, int incx, const TYPE *Y, int incy, TYPE *A, int lda) {       \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Uplo_=Uplo, N_=N, incx_=incx, incy_=incy, lda_=lda;     \
    TYPE alpha_=alpha;                                                          \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
              const TYPE*, const int*, const TYPE*, const int*,                 \
              TYPE*, const int*))                                                \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &N_, &alpha_,                                         \
         X, &incx_, Y, &incy_, A, &lda_);                                      \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Uplo, const int *N, const TYPE *alpha,         \
    const TYPE *X, const int *incx, const TYPE *Y, const int *incy,             \
    TYPE *A, const int *lda) {                                                  \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, int, const TYPE*, int,          \
              TYPE*, int))                                                       \
     g_cblas_fn[OP_ID])(*Order, *Uplo, *N, *alpha, X, *incx, Y, *incy, A, *lda); \
}

/** SYMV: (Order, Uplo, N, alpha, A, lda, X, incx, beta, Y, incy) */
#define DEFINE_SYMV_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Uplo, int N, TYPE alpha,                                     \
    const TYPE *A, int lda, const TYPE *X, int incx,                            \
    TYPE beta, TYPE *Y, int incy) {                                             \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Uplo_=Uplo, N_=N, lda_=lda, incx_=incx, incy_=incy;     \
    TYPE alpha_=alpha, beta_=beta;                                              \
    ((void(*)(const int*, const int*, const int*, const TYPE*,                  \
              const TYPE*, const int*, const TYPE*, const int*,                 \
              const TYPE*, TYPE*, const int*))                                   \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &N_, &alpha_,                                         \
         A, &lda_, X, &incx_, &beta_, Y, &incy_);                              \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Uplo, const int *N, const TYPE *alpha,         \
    const TYPE *A, const int *lda, const TYPE *X, const int *incx,              \
    const TYPE *beta, TYPE *Y, const int *incy) {                               \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, TYPE, const TYPE*, int, const TYPE*, int,          \
              TYPE, TYPE*, int))                                                 \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *Uplo, *N, *alpha, A, *lda, X, *incx, *beta, Y, *incy);       \
}

/** TRMV/TRSV: (Order, Uplo, Trans, Diag, N, A, lda, X, incx) */
#define DEFINE_TRMV_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Uplo, int Trans, int Diag, int N,                            \
    const TYPE *A, int lda, TYPE *X, int incx) {                                \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Uplo_=Uplo, Trans_=Trans, Diag_=Diag, N_=N;             \
    int lda_=lda, incx_=incx;                                                   \
    ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
              const TYPE*, const int*, TYPE*, const int*))                       \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &Trans_, &Diag_, &N_, A, &lda_, X, &incx_);          \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Uplo, const int *Trans, const int *Diag,       \
    const int *N, const TYPE *A, const int *lda, TYPE *X, const int *incx) {    \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int, const TYPE*, int, TYPE*, int))           \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *Uplo, *Trans, *Diag, *N, A, *lda, X, *incx);                 \
}

/* ============================================================================
 * Thunk helper macros — BLAS Level 3
 * ========================================================================== */

/** CGEMM/ZGEMM: alpha/beta are const void* (complex) */
#define DEFINE_CGEMM_THUNKS(OP, OP_ID)                                         \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int TransA, int TransB, int M, int N, int K,                     \
    const void *alpha, const void *A, int lda,                                  \
                       const void *B, int ldb,                                  \
    const void *beta, void *C, int ldc) {                                       \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, TransA_=TransA, TransB_=TransB;                          \
    int M_=M, N_=N, K_=K, lda_=lda, ldb_=ldb, ldc_=ldc;                       \
    ((void(*)(const int*, const int*, const int*,                               \
              const int*, const int*, const int*,                               \
              const void*, const void*, const int*,                              \
                           const void*, const int*,                              \
              const void*, void*, const int*))                                   \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &TransA_, &TransB_,                                           \
         &M_, &N_, &K_,                                                         \
         alpha, A, &lda_, B, &ldb_, beta, C, &ldc_);                           \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *TransA, const int *TransB,                     \
    const int *M, const int *N, const int *K,                                   \
    const void *alpha, const void *A, const int *lda,                           \
                       const void *B, const int *ldb,                           \
    const void *beta, void *C, const int *ldc) {                                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int, int,                                     \
              const void*, const void*, int, const void*, int,                  \
              const void*, void*, int))                                          \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *TransA, *TransB,                                              \
         *M, *N, *K,                                                            \
         alpha, A, *lda, B, *ldb, beta, C, *ldc);                              \
}

/** SYMM: (Order, Side, Uplo, M, N, alpha, A, lda, B, ldb, beta, C, ldc) */
#define DEFINE_SYMM_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Side, int Uplo, int M, int N,                                \
    TYPE alpha, const TYPE *A, int lda, const TYPE *B, int ldb,                 \
    TYPE beta, TYPE *C, int ldc) {                                              \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Side_=Side, Uplo_=Uplo, M_=M, N_=N;                     \
    TYPE alpha_=alpha, beta_=beta;                                              \
    int lda_=lda, ldb_=ldb, ldc_=ldc;                                          \
    ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
              const TYPE*, const TYPE*, const int*,                              \
                            const TYPE*, const int*,                             \
              const TYPE*, TYPE*, const int*))                                   \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Side_, &Uplo_, &M_, &N_,                                    \
         &alpha_, A, &lda_, B, &ldb_, &beta_, C, &ldc_);                       \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Side, const int *Uplo,                         \
    const int *M, const int *N,                                                 \
    const TYPE *alpha, const TYPE *A, const int *lda,                           \
                       const TYPE *B, const int *ldb,                           \
    const TYPE *beta, TYPE *C, const int *ldc) {                                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int,                                          \
              TYPE, const TYPE*, int, const TYPE*, int,                         \
              TYPE, TYPE*, int))                                                 \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *Side, *Uplo, *M, *N,                                         \
         *alpha, A, *lda, B, *ldb, *beta, C, *ldc);                            \
}

/** SYRK: (Order, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc) */
#define DEFINE_SYRK_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Uplo, int Trans, int N, int K,                               \
    TYPE alpha, const TYPE *A, int lda, TYPE beta, TYPE *C, int ldc) {          \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Uplo_=Uplo, Trans_=Trans, N_=N, K_=K;                   \
    TYPE alpha_=alpha, beta_=beta;                                              \
    int lda_=lda, ldc_=ldc;                                                     \
    ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
              const TYPE*, const TYPE*, const int*,                              \
              const TYPE*, TYPE*, const int*))                                   \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &Trans_, &N_, &K_,                                   \
         &alpha_, A, &lda_, &beta_, C, &ldc_);                                 \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Uplo, const int *Trans,                        \
    const int *N, const int *K,                                                 \
    const TYPE *alpha, const TYPE *A, const int *lda,                           \
    const TYPE *beta, TYPE *C, const int *ldc) {                                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int,                                          \
              TYPE, const TYPE*, int, TYPE, TYPE*, int))                         \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *Uplo, *Trans, *N, *K,                                        \
         *alpha, A, *lda, *beta, C, *ldc);                                     \
}

/** SYR2K: (Order, Uplo, Trans, N, K, alpha, A, lda, B, ldb, beta, C, ldc) */
#define DEFINE_SYR2K_THUNKS(OP, TYPE, OP_ID)                                   \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Uplo, int Trans, int N, int K,                               \
    TYPE alpha, const TYPE *A, int lda, const TYPE *B, int ldb,                 \
    TYPE beta, TYPE *C, int ldc) {                                              \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Uplo_=Uplo, Trans_=Trans, N_=N, K_=K;                   \
    TYPE alpha_=alpha, beta_=beta;                                              \
    int lda_=lda, ldb_=ldb, ldc_=ldc;                                          \
    ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
              const TYPE*, const TYPE*, const int*,                              \
                            const TYPE*, const int*,                             \
              const TYPE*, TYPE*, const int*))                                   \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Uplo_, &Trans_, &N_, &K_,                                   \
         &alpha_, A, &lda_, B, &ldb_, &beta_, C, &ldc_);                       \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Uplo, const int *Trans,                        \
    const int *N, const int *K,                                                 \
    const TYPE *alpha, const TYPE *A, const int *lda,                           \
                       const TYPE *B, const int *ldb,                           \
    const TYPE *beta, TYPE *C, const int *ldc) {                                \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int,                                          \
              TYPE, const TYPE*, int, const TYPE*, int,                         \
              TYPE, TYPE*, int))                                                 \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *Uplo, *Trans, *N, *K,                                        \
         *alpha, A, *lda, B, *ldb, *beta, C, *ldc);                            \
}

/** TRMM/TRSM: (Order, Side, Uplo, Trans, Diag, M, N, alpha, A, lda, B, ldb) */
#define DEFINE_TRMM_THUNKS(OP, TYPE, OP_ID)                                    \
static void thunk_##OP##_cblas_to_fortran(                                      \
    int Order, int Side, int Uplo, int Trans, int Diag,                         \
    int M, int N, TYPE alpha, const TYPE *A, int lda, TYPE *B, int ldb) {       \
    if (!g_fortran_fn[OP_ID]) return;                                           \
    int Order_=Order, Side_=Side, Uplo_=Uplo, Trans_=Trans, Diag_=Diag;       \
    int M_=M, N_=N, lda_=lda, ldb_=ldb;                                        \
    TYPE alpha_=alpha;                                                          \
    ((void(*)(const int*, const int*, const int*, const int*, const int*,       \
              const int*, const int*, const TYPE*,                               \
              const TYPE*, const int*, TYPE*, const int*))                       \
     g_fortran_fn[OP_ID])(                                                      \
         &Order_, &Side_, &Uplo_, &Trans_, &Diag_,                             \
         &M_, &N_, &alpha_, A, &lda_, B, &ldb_);                               \
}                                                                               \
static void thunk_##OP##_fortran_to_cblas(                                      \
    const int *Order, const int *Side, const int *Uplo,                         \
    const int *Trans, const int *Diag,                                          \
    const int *M, const int *N, const TYPE *alpha,                              \
    const TYPE *A, const int *lda, TYPE *B, const int *ldb) {                   \
    if (!g_cblas_fn[OP_ID]) return;                                             \
    ((void(*)(int, int, int, int, int, int, int,                                \
              TYPE, const TYPE*, int, TYPE*, int))                               \
     g_cblas_fn[OP_ID])(                                                        \
         *Order, *Side, *Uplo, *Trans, *Diag,                                  \
         *M, *N, *alpha, A, *lda, B, *ldb);                                    \
}

/* ============================================================================
 * Thunk instantiation — BLAS Level 1 completion (rot, iamax)
 * ========================================================================== */

DEFINE_ROT_THUNKS(srot, float,  FB_OP_SROT)
DEFINE_ROT_THUNKS(drot, double, FB_OP_DROT)
DEFINE_IAMAX_THUNKS(isamax, float,  FB_OP_ISAMAX)
DEFINE_IAMAX_THUNKS(idamax, double, FB_OP_IDAMAX)

/* ============================================================================
 * Thunk instantiation — BLAS Level 2
 * ========================================================================== */

DEFINE_GEMV_THUNKS(sgemv, float,  FB_OP_SGEMV)
DEFINE_GEMV_THUNKS(dgemv, double, FB_OP_DGEMV)
DEFINE_GER_THUNKS(sger,   float,  FB_OP_SGER)
DEFINE_GER_THUNKS(dger,   double, FB_OP_DGER)
DEFINE_SYR_THUNKS(ssyr,   float,  FB_OP_SSYR)
DEFINE_SYR_THUNKS(dsyr,   double, FB_OP_DSYR)
DEFINE_SYR2_THUNKS(ssyr2, float,  FB_OP_SSYR2)
DEFINE_SYR2_THUNKS(dsyr2, double, FB_OP_DSYR2)
DEFINE_SYMV_THUNKS(ssymv, float,  FB_OP_SSYMV)
DEFINE_SYMV_THUNKS(dsymv, double, FB_OP_DSYMV)
DEFINE_TRMV_THUNKS(strmv, float,  FB_OP_STRMV)
DEFINE_TRMV_THUNKS(dtrmv, double, FB_OP_DTRMV)
DEFINE_TRMV_THUNKS(strsv, float,  FB_OP_STRSV)
DEFINE_TRMV_THUNKS(dtrsv, double, FB_OP_DTRSV)

/* ============================================================================
 * Thunk instantiation — BLAS Level 3 completion (complex + L3 variants)
 * ========================================================================== */

DEFINE_CGEMM_THUNKS(cgemm, FB_OP_CGEMM)
DEFINE_CGEMM_THUNKS(zgemm, FB_OP_ZGEMM)
DEFINE_SYMM_THUNKS(ssymm,   float,  FB_OP_SSYMM)
DEFINE_SYMM_THUNKS(dsymm,   double, FB_OP_DSYMM)
DEFINE_SYRK_THUNKS(ssyrk,   float,  FB_OP_SSYRK)
DEFINE_SYRK_THUNKS(dsyrk,   double, FB_OP_DSYRK)
DEFINE_SYR2K_THUNKS(ssyr2k, float,  FB_OP_SSYR2K)
DEFINE_SYR2K_THUNKS(dsyr2k, double, FB_OP_DSYR2K)
DEFINE_TRMM_THUNKS(strmm,   float,  FB_OP_STRMM)
DEFINE_TRMM_THUNKS(dtrmm,   double, FB_OP_DTRMM)
DEFINE_TRMM_THUNKS(strsm,   float,  FB_OP_STRSM)
DEFINE_TRMM_THUNKS(dtrsm,   double, FB_OP_DTRSM)

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
    /* Level 1 */
    [FB_OP_SAXPY]   = C2F(thunk_saxpy_cblas_to_fortran),
    [FB_OP_DAXPY]   = C2F(thunk_daxpy_cblas_to_fortran),
    [FB_OP_CAXPY]   = C2F(thunk_caxpy_cblas_to_fortran),
    [FB_OP_ZAXPY]   = C2F(thunk_zaxpy_cblas_to_fortran),
    [FB_OP_SSCAL]   = C2F(thunk_sscal_cblas_to_fortran),
    [FB_OP_DSCAL]   = C2F(thunk_dscal_cblas_to_fortran),
    [FB_OP_SCOPY]   = C2F(thunk_scopy_cblas_to_fortran),
    [FB_OP_DCOPY]   = C2F(thunk_dcopy_cblas_to_fortran),
    [FB_OP_SSWAP]   = C2F(thunk_sswap_cblas_to_fortran),
    [FB_OP_DSWAP]   = C2F(thunk_dswap_cblas_to_fortran),
    [FB_OP_SNRM2]   = C2F(thunk_snrm2_cblas_to_fortran),
    [FB_OP_DNRM2]   = C2F(thunk_dnrm2_cblas_to_fortran),
    [FB_OP_SCNRM2]  = C2F(thunk_scnrm2_cblas_to_fortran),
    [FB_OP_DZNRM2]  = C2F(thunk_dznrm2_cblas_to_fortran),
    [FB_OP_SASUM]   = C2F(thunk_sasum_cblas_to_fortran),
    [FB_OP_DASUM]   = C2F(thunk_dasum_cblas_to_fortran),
    [FB_OP_SCASUM]  = C2F(thunk_scasum_cblas_to_fortran),
    [FB_OP_DZASUM]  = C2F(thunk_dzasum_cblas_to_fortran),
    [FB_OP_SDOT]    = C2F(thunk_sdot_cblas_to_fortran),
    [FB_OP_DDOT]    = C2F(thunk_ddot_cblas_to_fortran),
    [FB_OP_SROT]    = C2F(thunk_srot_cblas_to_fortran),
    [FB_OP_DROT]    = C2F(thunk_drot_cblas_to_fortran),
    [FB_OP_ISAMAX]  = C2F(thunk_isamax_cblas_to_fortran),
    [FB_OP_IDAMAX]  = C2F(thunk_idamax_cblas_to_fortran),
    /* Level 2 */
    [FB_OP_SGEMV]   = C2F(thunk_sgemv_cblas_to_fortran),
    [FB_OP_DGEMV]   = C2F(thunk_dgemv_cblas_to_fortran),
    [FB_OP_SGER]    = C2F(thunk_sger_cblas_to_fortran),
    [FB_OP_DGER]    = C2F(thunk_dger_cblas_to_fortran),
    [FB_OP_SSYR]    = C2F(thunk_ssyr_cblas_to_fortran),
    [FB_OP_DSYR]    = C2F(thunk_dsyr_cblas_to_fortran),
    [FB_OP_SSYR2]   = C2F(thunk_ssyr2_cblas_to_fortran),
    [FB_OP_DSYR2]   = C2F(thunk_dsyr2_cblas_to_fortran),
    [FB_OP_SSYMV]   = C2F(thunk_ssymv_cblas_to_fortran),
    [FB_OP_DSYMV]   = C2F(thunk_dsymv_cblas_to_fortran),
    [FB_OP_STRMV]   = C2F(thunk_strmv_cblas_to_fortran),
    [FB_OP_DTRMV]   = C2F(thunk_dtrmv_cblas_to_fortran),
    [FB_OP_STRSV]   = C2F(thunk_strsv_cblas_to_fortran),
    [FB_OP_DTRSV]   = C2F(thunk_dtrsv_cblas_to_fortran),
    /* Level 3 */
    [FB_OP_SGEMM]   = C2F(thunk_sgemm_cblas_to_fortran),
    [FB_OP_DGEMM]   = C2F(thunk_dgemm_cblas_to_fortran),
    [FB_OP_CGEMM]   = C2F(thunk_cgemm_cblas_to_fortran),
    [FB_OP_ZGEMM]   = C2F(thunk_zgemm_cblas_to_fortran),
    [FB_OP_SSYMM]   = C2F(thunk_ssymm_cblas_to_fortran),
    [FB_OP_DSYMM]   = C2F(thunk_dsymm_cblas_to_fortran),
    [FB_OP_SSYRK]   = C2F(thunk_ssyrk_cblas_to_fortran),
    [FB_OP_DSYRK]   = C2F(thunk_dsyrk_cblas_to_fortran),
    [FB_OP_SSYR2K]  = C2F(thunk_ssyr2k_cblas_to_fortran),
    [FB_OP_DSYR2K]  = C2F(thunk_dsyr2k_cblas_to_fortran),
    [FB_OP_STRMM]   = C2F(thunk_strmm_cblas_to_fortran),
    [FB_OP_DTRMM]   = C2F(thunk_dtrmm_cblas_to_fortran),
    [FB_OP_STRSM]   = C2F(thunk_strsm_cblas_to_fortran),
    [FB_OP_DTRSM]   = C2F(thunk_dtrsm_cblas_to_fortran),
};

fb_generic_fn const k_fortran_to_cblas_thunks[FB_JUDGE_MAX_OPERATIONS] = {
    /* Level 1 */
    [FB_OP_SAXPY]   = F2C(thunk_saxpy_fortran_to_cblas),
    [FB_OP_DAXPY]   = F2C(thunk_daxpy_fortran_to_cblas),
    [FB_OP_CAXPY]   = F2C(thunk_caxpy_fortran_to_cblas),
    [FB_OP_ZAXPY]   = F2C(thunk_zaxpy_fortran_to_cblas),
    [FB_OP_SSCAL]   = F2C(thunk_sscal_fortran_to_cblas),
    [FB_OP_DSCAL]   = F2C(thunk_dscal_fortran_to_cblas),
    [FB_OP_SCOPY]   = F2C(thunk_scopy_fortran_to_cblas),
    [FB_OP_DCOPY]   = F2C(thunk_dcopy_fortran_to_cblas),
    [FB_OP_SSWAP]   = F2C(thunk_sswap_fortran_to_cblas),
    [FB_OP_DSWAP]   = F2C(thunk_dswap_fortran_to_cblas),
    [FB_OP_SNRM2]   = F2C(thunk_snrm2_fortran_to_cblas),
    [FB_OP_DNRM2]   = F2C(thunk_dnrm2_fortran_to_cblas),
    [FB_OP_SCNRM2]  = F2C(thunk_scnrm2_fortran_to_cblas),
    [FB_OP_DZNRM2]  = F2C(thunk_dznrm2_fortran_to_cblas),
    [FB_OP_SASUM]   = F2C(thunk_sasum_fortran_to_cblas),
    [FB_OP_DASUM]   = F2C(thunk_dasum_fortran_to_cblas),
    [FB_OP_SCASUM]  = F2C(thunk_scasum_fortran_to_cblas),
    [FB_OP_DZASUM]  = F2C(thunk_dzasum_fortran_to_cblas),
    [FB_OP_SDOT]    = F2C(thunk_sdot_fortran_to_cblas),
    [FB_OP_DDOT]    = F2C(thunk_ddot_fortran_to_cblas),
    [FB_OP_SROT]    = F2C(thunk_srot_fortran_to_cblas),
    [FB_OP_DROT]    = F2C(thunk_drot_fortran_to_cblas),
    [FB_OP_ISAMAX]  = F2C(thunk_isamax_fortran_to_cblas),
    [FB_OP_IDAMAX]  = F2C(thunk_idamax_fortran_to_cblas),
    /* Level 2 */
    [FB_OP_SGEMV]   = F2C(thunk_sgemv_fortran_to_cblas),
    [FB_OP_DGEMV]   = F2C(thunk_dgemv_fortran_to_cblas),
    [FB_OP_SGER]    = F2C(thunk_sger_fortran_to_cblas),
    [FB_OP_DGER]    = F2C(thunk_dger_fortran_to_cblas),
    [FB_OP_SSYR]    = F2C(thunk_ssyr_fortran_to_cblas),
    [FB_OP_DSYR]    = F2C(thunk_dsyr_fortran_to_cblas),
    [FB_OP_SSYR2]   = F2C(thunk_ssyr2_fortran_to_cblas),
    [FB_OP_DSYR2]   = F2C(thunk_dsyr2_fortran_to_cblas),
    [FB_OP_SSYMV]   = F2C(thunk_ssymv_fortran_to_cblas),
    [FB_OP_DSYMV]   = F2C(thunk_dsymv_fortran_to_cblas),
    [FB_OP_STRMV]   = F2C(thunk_strmv_fortran_to_cblas),
    [FB_OP_DTRMV]   = F2C(thunk_dtrmv_fortran_to_cblas),
    [FB_OP_STRSV]   = F2C(thunk_strsv_fortran_to_cblas),
    [FB_OP_DTRSV]   = F2C(thunk_dtrsv_fortran_to_cblas),
    /* Level 3 */
    [FB_OP_SGEMM]   = F2C(thunk_sgemm_fortran_to_cblas),
    [FB_OP_DGEMM]   = F2C(thunk_dgemm_fortran_to_cblas),
    [FB_OP_CGEMM]   = F2C(thunk_cgemm_fortran_to_cblas),
    [FB_OP_ZGEMM]   = F2C(thunk_zgemm_fortran_to_cblas),
    [FB_OP_SSYMM]   = F2C(thunk_ssymm_fortran_to_cblas),
    [FB_OP_DSYMM]   = F2C(thunk_dsymm_fortran_to_cblas),
    [FB_OP_SSYRK]   = F2C(thunk_ssyrk_fortran_to_cblas),
    [FB_OP_DSYRK]   = F2C(thunk_dsyrk_fortran_to_cblas),
    [FB_OP_SSYR2K]  = F2C(thunk_ssyr2k_fortran_to_cblas),
    [FB_OP_DSYR2K]  = F2C(thunk_dsyr2k_fortran_to_cblas),
    [FB_OP_STRMM]   = F2C(thunk_strmm_fortran_to_cblas),
    [FB_OP_DTRMM]   = F2C(thunk_dtrmm_fortran_to_cblas),
    [FB_OP_STRSM]   = F2C(thunk_strsm_fortran_to_cblas),
    [FB_OP_DTRSM]   = F2C(thunk_dtrsm_fortran_to_cblas),
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

}
