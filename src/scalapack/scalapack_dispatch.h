/**
 * @file scalapack_dispatch.h
 * @brief Internal ScaLAPACK dispatch — Fortran-ABI shims and op-id router.
 *
 * Not part of the public API.  Include <faster-blaster/scalapack.h> instead.
 *
 * @copyright Copyright (c) 2025
 * @license MIT OR Apache-2.0
 */

#ifndef FB_SCALAPACK_DISPATCH_H
#define FB_SCALAPACK_DISPATCH_H

#include <stdbool.h>
#include <stdint.h>
#include "../../include/faster-blaster/scalapack.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ─────────────────────────────────────────────────────────────────────────
 * Fortran ScaLAPACK / BLACS function pointer typedefs
 *
 * Every argument is pass-by-pointer per the Fortran calling convention.
 * Character arguments are passed as pointer + hidden length (platform-specific;
 * gfortran/ifort compatible layout used here: length after all other args).
 * ───────────────────────────────────────────────────────────────────────── */

/* BLACS lifecycle */
typedef void (*blacs_pinfo_t)   (int *mypnum, int *nprocs);
typedef void (*blacs_get_t)     (int *ictxt,  int *what, int *val);
typedef void (*blacs_gridinit_t)(int *ictxt,  const char *order, int order_len,
                                 int *nprow,  int *npcol);
typedef void (*blacs_gridinfo_t)(int *ictxt,  int *nprow, int *npcol,
                                 int *myrow,  int *mycol);
typedef void (*blacs_gridexit_t)(int *ictxt);

/* PBLAS / ScaLAPACK Level 1 */
typedef void (*psaxpy_t)(int *n, float  *alpha,
                         const float  *X, int *ix, int *jx, const int *descx, int *incx,
                         float  *Y, int *iy, int *jy, const int *descy, int *incy);
typedef void (*pdaxpy_t)(int *n, double *alpha,
                         const double *X, int *ix, int *jx, const int *descx, int *incx,
                         double *Y, int *iy, int *jy, const int *descy, int *incy);

/* PBLAS Level 3 */
typedef void (*psgemm_t)(const char *transa, const char *transb,
                         int *m, int *n, int *k, float *alpha,
                         const float *A, int *ia, int *ja, const int *desca,
                         const float *B, int *ib, int *jb, const int *descb,
                         float *beta, float *C, int *ic, int *jc, const int *descc,
                         int transa_len, int transb_len);
typedef void (*pdgemm_t)(const char *transa, const char *transb,
                         int *m, int *n, int *k, double *alpha,
                         const double *A, int *ia, int *ja, const int *desca,
                         const double *B, int *ib, int *jb, const int *descb,
                         double *beta, double *C, int *ic, int *jc, const int *descc,
                         int transa_len, int transb_len);
typedef void (*pstrsm_t)(const char *side, const char *uplo,
                         const char *transa, const char *diag,
                         int *m, int *n, float *alpha,
                         const float *A, int *ia, int *ja, const int *desca,
                         float *B, int *ib, int *jb, const int *descb,
                         int side_len, int uplo_len, int transa_len, int diag_len);
typedef void (*pdtrsm_t)(const char *side, const char *uplo,
                         const char *transa, const char *diag,
                         int *m, int *n, double *alpha,
                         const double *A, int *ia, int *ja, const int *desca,
                         double *B, int *ib, int *jb, const int *descb,
                         int side_len, int uplo_len, int transa_len, int diag_len);

/* ScaLAPACK drivers */
typedef void (*psgetrf_t)(int *m, int *n,
                          float *A, int *ia, int *ja, const int *desca,
                          int *ipiv, int *info);
typedef void (*pdgetrf_t)(int *m, int *n,
                          double *A, int *ia, int *ja, const int *desca,
                          int *ipiv, int *info);
typedef void (*psgetrs_t)(const char *trans, int *n, int *nrhs,
                          const float *A, int *ia, int *ja, const int *desca,
                          const int *ipiv,
                          float *B, int *ib, int *jb, const int *descb,
                          int *info, int trans_len);
typedef void (*pdgetrs_t)(const char *trans, int *n, int *nrhs,
                          const double *A, int *ia, int *ja, const int *desca,
                          const int *ipiv,
                          double *B, int *ib, int *jb, const int *descb,
                          int *info, int trans_len);
typedef void (*psgesv_t) (int *n, int *nrhs,
                          float *A, int *ia, int *ja, const int *desca,
                          int *ipiv,
                          float *B, int *ib, int *jb, const int *descb,
                          int *info);
typedef void (*pdgesv_t) (int *n, int *nrhs,
                          double *A, int *ia, int *ja, const int *desca,
                          int *ipiv,
                          double *B, int *ib, int *jb, const int *descb,
                          int *info);
typedef void (*pspotrf_t)(const char *uplo, int *n,
                          float *A, int *ia, int *ja, const int *desca,
                          int *info, int uplo_len);
typedef void (*pdpotrf_t)(const char *uplo, int *n,
                          double *A, int *ia, int *ja, const int *desca,
                          int *info, int uplo_len);
typedef void (*psgeqrf_t)(int *m, int *n,
                          float *A, int *ia, int *ja, const int *desca,
                          float *tau, float *work, int *lwork, int *info);
typedef void (*pdgeqrf_t)(int *m, int *n,
                          double *A, int *ia, int *ja, const int *desca,
                          double *tau, double *work, int *lwork, int *info);
typedef void (*psgesvd_t)(const char *jobu, const char *jobvt,
                          int *m, int *n,
                          float *A, int *ia, int *ja, const int *desca,
                          float *S,
                          float *U,  int *iu,  int *ju,  const int *descu,
                          float *VT, int *ivt, int *jvt, const int *descvt,
                          float *work, int *lwork, int *info,
                          int jobu_len, int jobvt_len);
typedef void (*pdgesvd_t)(const char *jobu, const char *jobvt,
                          int *m, int *n,
                          double *A, int *ia, int *ja, const int *desca,
                          double *S,
                          double *U,  int *iu,  int *ju,  const int *descu,
                          double *VT, int *ivt, int *jvt, const int *descvt,
                          double *work, int *lwork, int *info,
                          int jobu_len, int jobvt_len);

/* ─────────────────────────────────────────────────────────────────────────
 * Internal loader
 * ───────────────────────────────────────────────────────────────────────── */

/**
 * Load ScaLAPACK and BLACS shared libraries.
 * Returns 0 on success, -1 if the libraries cannot be found.
 * Idempotent — safe to call multiple times.
 */
int fb_scalapack_load(void);

#ifdef __cplusplus
}
#endif

#endif /* FB_SCALAPACK_DISPATCH_H */
