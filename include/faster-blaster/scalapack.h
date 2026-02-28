/**
 * @file scalapack.h
 * @brief Public ScaLAPACK dispatch API for faster-blaster.
 *
 * ScaLAPACK routines distribute matrix computation across an MPI process grid
 * managed by BLACS.  This header exposes:
 *
 *  1. BLACS process-grid lifecycle  (fb_distributed_ctx_init / _shutdown)
 *  2. Array descriptor helper       (fb_scalapack_desc_t, fb_scalapack_desc_init)
 *  3. C-style wrappers around the most common ScaLAPACK routines, whose
 *     underlying Fortran symbols are loaded dynamically at runtime — no
 *     ScaLAPACK headers or link-time libraries are required.
 *
 * Integration with the judge/sequence executor
 * ─────────────────────────────────────────────
 * All ScaLAPACK op IDs (FB_OP_PSGEMM … FB_OP__SCALAPACK_END) live in the same
 * flat FB_OP_* namespace as BLAS/LAPACK.  The sequence executor checks:
 *
 *     if (op_id >= FB_OP_SCALAPACK_BASE && op_id < FB_OP_SCALAPACK_END)
 *         fb_scalapack_dispatch_op(op_id, exec_ctx->dist, args);
 *
 * so mixed BLAS + ScaLAPACK sequences are planned and reasoned over by the
 * judge as one unified op stream.
 *
 * @note MPI_Comm is accepted as `void*` to avoid dragging in mpi.h.
 *       Cast MPI_COMM_WORLD (or any MPI_Comm) to void* at the call site.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FASTER_BLASTER_SCALAPACK_H
#define FASTER_BLASTER_SCALAPACK_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─────────────────────────────────────────────────────────────────────────
 * ScaLAPACK op-ID range constants
 * ───────────────────────────────────────────────────────────────────────── */

/** First FB_OP_* value belonging to the ScaLAPACK namespace. */
#define FB_OP_SCALAPACK_BASE  1322u

/** One past the last FB_OP_* value in the ScaLAPACK namespace. */
#define FB_OP_SCALAPACK_END   1635u

/** Returns true if @p op_id is a ScaLAPACK operation. */
static inline bool fb_scalapack_op_in_range(uint32_t op_id) {
    return op_id >= FB_OP_SCALAPACK_BASE && op_id < FB_OP_SCALAPACK_END;
}

/* ─────────────────────────────────────────────────────────────────────────
 * BLACS process-grid context
 * ───────────────────────────────────────────────────────────────────────── */

/**
 * @brief BLACS process-grid descriptor.
 *
 * Populated by fb_distributed_ctx_init().  Callers should treat this as
 * read-only after initialization.
 */
typedef struct fb_blacs_ctx {
    int ictxt;      /**< BLACS context handle (opaque integer)         */
    int nprow;      /**< Number of process rows in the grid            */
    int npcol;      /**< Number of process columns in the grid         */
    int myrow;      /**< Row index of this process (0-based)           */
    int mycol;      /**< Column index of this process (0-based)        */
    int mypnum;     /**< Global MPI rank of this process               */
    int nprocs;     /**< Total number of MPI processes                 */
} fb_blacs_ctx_t;

/**
 * @brief Distributed (BLACS/MPI) execution context.
 *
 * Created once per MPI communicator + grid shape.  Pass a pointer to this
 * inside fb_exec_context_t.dist when submitting sequences that include
 * ScaLAPACK operations.
 */
typedef struct fb_distributed_ctx {
    fb_blacs_ctx_t blacs;       /**< Process-grid metadata              */
    void          *mpi_comm;    /**< Opaque MPI_Comm (cast from MPI_Comm) */
    bool           initialized; /**< True after successful init         */
} fb_distributed_ctx_t;

/**
 * @brief Initialize a distributed context.
 *
 * Calls BLACS to set up a @p nprow × @p npcol process grid over
 * @p mpi_comm.  The calling process must have already called MPI_Init.
 *
 * @param ctx        Output context to populate.
 * @param mpi_comm   MPI communicator cast to void* (e.g. (void*)MPI_COMM_WORLD).
 * @param nprow      Process grid row count.
 * @param npcol      Process grid column count.
 * @return 0 on success; -1 if ScaLAPACK/BLACS not available; -2 on BLACS error.
 */
int fb_distributed_ctx_init(fb_distributed_ctx_t *ctx,
                             void *mpi_comm,
                             int   nprow,
                             int   npcol);

/**
 * @brief Shutdown a distributed context, releasing the BLACS grid.
 * @param ctx Context previously initialized with fb_distributed_ctx_init().
 */
void fb_distributed_ctx_shutdown(fb_distributed_ctx_t *ctx);

/**
 * @brief Returns true if a ScaLAPACK library was successfully loaded.
 */
bool fb_scalapack_is_available(void);

/* ─────────────────────────────────────────────────────────────────────────
 * ScaLAPACK array descriptor
 * ───────────────────────────────────────────────────────────────────────── */

/**
 * @brief ScaLAPACK block-cyclic 2D array descriptor (9 integers).
 *
 * DESCA layout (ScaLAPACK convention):
 *   [0] dtype  = 1   (block-cyclic 2D)
 *   [1] ctxt        (BLACS context)
 *   [2] m           (global row count)
 *   [3] n           (global col count)
 *   [4] mb          (row block size)
 *   [5] nb          (col block size)
 *   [6] rsrc   = 0  (first row process)
 *   [7] csrc   = 0  (first col process)
 *   [8] lld         (local leading dimension)
 */
typedef int fb_scalapack_desc_t[9];

/**
 * @brief Populate a ScaLAPACK block-cyclic descriptor.
 *
 * @param desc  Output descriptor array (9 ints).
 * @param ctx   Initialized distributed context.
 * @param m     Global number of rows.
 * @param n     Global number of columns.
 * @param mb    Row block size.
 * @param nb    Column block size.
 * @param lld   Local leading dimension (≥ local row count).
 */
void fb_scalapack_desc_init(fb_scalapack_desc_t desc,
                             const fb_distributed_ctx_t *ctx,
                             int m, int n,
                             int mb, int nb,
                             int lld);

/* ─────────────────────────────────────────────────────────────────────────
 * C-style public wrappers  (ScaLAPACK Level 1)
 * ───────────────────────────────────────────────────────────────────────── */

/** Parallel single-precision AXPY: sub(Y) := alpha*sub(X) + sub(Y) */
void fb_psaxpy(int n, float alpha,
               const float *X, int ix, int jx, const fb_scalapack_desc_t descx, int incx,
               float *Y, int iy, int jy, const fb_scalapack_desc_t descy, int incy,
               const fb_distributed_ctx_t *ctx);

/** Parallel double-precision AXPY */
void fb_pdaxpy(int n, double alpha,
               const double *X, int ix, int jx, const fb_scalapack_desc_t descx, int incx,
               double *Y, int iy, int jy, const fb_scalapack_desc_t descy, int incy,
               const fb_distributed_ctx_t *ctx);

/* ─────────────────────────────────────────────────────────────────────────
 * C-style public wrappers  (ScaLAPACK Level 3 / PBLAS)
 * ───────────────────────────────────────────────────────────────────────── */

/**
 * @brief Parallel single-precision GEMM: sub(C) := alpha*op(A)*op(B) + beta*sub(C)
 */
void fb_psgemm(char transa, char transb,
               int m, int n, int k,
               float alpha,
               const float *A, int ia, int ja, const fb_scalapack_desc_t desca,
               const float *B, int ib, int jb, const fb_scalapack_desc_t descb,
               float beta,
               float *C, int ic, int jc, const fb_scalapack_desc_t descc,
               const fb_distributed_ctx_t *ctx);

/** Parallel double-precision GEMM */
void fb_pdgemm(char transa, char transb,
               int m, int n, int k,
               double alpha,
               const double *A, int ia, int ja, const fb_scalapack_desc_t desca,
               const double *B, int ib, int jb, const fb_scalapack_desc_t descb,
               double beta,
               double *C, int ic, int jc, const fb_scalapack_desc_t descc,
               const fb_distributed_ctx_t *ctx);

/** Parallel single-precision TRSM */
void fb_pstrsm(char side, char uplo, char transa, char diag,
               int m, int n, float alpha,
               const float *A, int ia, int ja, const fb_scalapack_desc_t desca,
               float *B, int ib, int jb, const fb_scalapack_desc_t descb,
               const fb_distributed_ctx_t *ctx);

/** Parallel double-precision TRSM */
void fb_pdtrsm(char side, char uplo, char transa, char diag,
               int m, int n, double alpha,
               const double *A, int ia, int ja, const fb_scalapack_desc_t desca,
               double *B, int ib, int jb, const fb_scalapack_desc_t descb,
               const fb_distributed_ctx_t *ctx);

/* ─────────────────────────────────────────────────────────────────────────
 * C-style public wrappers  (ScaLAPACK drivers)
 * ───────────────────────────────────────────────────────────────────────── */

/**
 * @brief Parallel single-precision LU factorization: sub(A) := P*L*U
 * @param[out] ipiv  Integer pivot array (local, sized numroc(m, mb, myrow, 0, nprow) + mb)
 * @param[out] info  0 = success; >0 = singular at that position
 */
void fb_psgetrf(int m, int n,
                float *A, int ia, int ja, const fb_scalapack_desc_t desca,
                int *ipiv, int *info,
                const fb_distributed_ctx_t *ctx);

/** Parallel double-precision LU factorization */
void fb_pdgetrf(int m, int n,
                double *A, int ia, int ja, const fb_scalapack_desc_t desca,
                int *ipiv, int *info,
                const fb_distributed_ctx_t *ctx);

/** Parallel single-precision solve using LU factorization from fb_psgetrf */
void fb_psgetrs(char trans, int n, int nrhs,
                const float *A, int ia, int ja, const fb_scalapack_desc_t desca,
                const int *ipiv,
                float *B, int ib, int jb, const fb_scalapack_desc_t descb,
                int *info,
                const fb_distributed_ctx_t *ctx);

/** Parallel double-precision solve using LU factorization */
void fb_pdgetrs(char trans, int n, int nrhs,
                const double *A, int ia, int ja, const fb_scalapack_desc_t desca,
                const int *ipiv,
                double *B, int ib, int jb, const fb_scalapack_desc_t descb,
                int *info,
                const fb_distributed_ctx_t *ctx);

/** Parallel single-precision general linear solve (driver: gesv = getrf + getrs) */
void fb_psgesv(int n, int nrhs,
               float *A, int ia, int ja, const fb_scalapack_desc_t desca,
               int *ipiv,
               float *B, int ib, int jb, const fb_scalapack_desc_t descb,
               int *info,
               const fb_distributed_ctx_t *ctx);

/** Parallel double-precision general linear solve */
void fb_pdgesv(int n, int nrhs,
               double *A, int ia, int ja, const fb_scalapack_desc_t desca,
               int *ipiv,
               double *B, int ib, int jb, const fb_scalapack_desc_t descb,
               int *info,
               const fb_distributed_ctx_t *ctx);

/** Parallel single-precision Cholesky factorization */
void fb_pspotrf(char uplo, int n,
                float *A, int ia, int ja, const fb_scalapack_desc_t desca,
                int *info,
                const fb_distributed_ctx_t *ctx);

/** Parallel double-precision Cholesky factorization */
void fb_pdpotrf(char uplo, int n,
                double *A, int ia, int ja, const fb_scalapack_desc_t desca,
                int *info,
                const fb_distributed_ctx_t *ctx);

/** Parallel single-precision QR factorization */
void fb_psgeqrf(int m, int n,
                float *A, int ia, int ja, const fb_scalapack_desc_t desca,
                float *tau, float *work, int lwork,
                int *info,
                const fb_distributed_ctx_t *ctx);

/** Parallel double-precision QR factorization */
void fb_pdgeqrf(int m, int n,
                double *A, int ia, int ja, const fb_scalapack_desc_t desca,
                double *tau, double *work, int lwork,
                int *info,
                const fb_distributed_ctx_t *ctx);

/** Parallel single-precision SVD */
void fb_psgesvd(char jobu, char jobvt,
                int m, int n,
                float *A, int ia, int ja, const fb_scalapack_desc_t desca,
                float *S,
                float *U,  int iu,  int ju,  const fb_scalapack_desc_t descu,
                float *VT, int ivt, int jvt, const fb_scalapack_desc_t descvt,
                float *work, int lwork,
                int *info,
                const fb_distributed_ctx_t *ctx);

/** Parallel double-precision SVD */
void fb_pdgesvd(char jobu, char jobvt,
                int m, int n,
                double *A, int ia, int ja, const fb_scalapack_desc_t desca,
                double *S,
                double *U,  int iu,  int ju,  const fb_scalapack_desc_t descu,
                double *VT, int ivt, int jvt, const fb_scalapack_desc_t descvt,
                double *work, int lwork,
                int *info,
                const fb_distributed_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif /* FASTER_BLASTER_SCALAPACK_H */
