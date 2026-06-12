#ifndef BLAS_LAPACK_REFERENCE_COMPLETE_H
#define BLAS_LAPACK_REFERENCE_COMPLETE_H

#include <stddef.h>
#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
   BLAS LEVEL 1: Vector Operations (56 operations)
   ============================================================================ */

/* DOT - Dot product */
float sdot_(const int *n, const float *x, const int *incx, const float *y, const int *incy);
double ddot_(const int *n, const double *x, const int *incx, const double *y, const int *incy);
float cdotu_(const int *n, const void *x, const int *incx, const void *y, const int *incy);
void cdotc_(void *ret, const int *n, const void *x, const int *incx, const void *y, const int *incy);
double zdotu_(const int *n, const void *x, const int *incx, const void *y, const int *incy);
void zdotc_(void *ret, const int *n, const void *x, const int *incx, const void *y, const int *incy);

/* AXPY - Y := A*X + Y */
void saxpy_(const int *n, const float *a, const float *x, const int *incx, float *y, const int *incy);
void daxpy_(const int *n, const double *a, const double *x, const int *incx, double *y, const int *incy);
void caxpy_(const int *n, const void *a, const void *x, const int *incx, void *y, const int *incy);
void zaxpy_(const int *n, const void *a, const void *x, const int *incx, void *y, const int *incy);

/* COPY - Y := X */
void scopy_(const int *n, const float *x, const int *incx, float *y, const int *incy);
void dcopy_(const int *n, const double *x, const int *incx, double *y, const int *incy);
void ccopy_(const int *n, const void *x, const int *incx, void *y, const int *incy);
void zcopy_(const int *n, const void *x, const int *incx, void *y, const int *incy);

/* SCAL - X := A*X */
void sscal_(const int *n, const float *a, float *x, const int *incx);
void dscal_(const int *n, const double *a, double *x, const int *incx);
void cscal_(const int *n, const void *a, void *x, const int *incx);
void zscal_(const int *n, const void *a, void *x, const int *incx);
void csscal_(const int *n, const float *a, void *x, const int *incx);
void zdscal_(const int *n, const double *a, void *x, const int *incx);

/* NRM2 - 2-norm of X */
float snrm2_(const int *n, const float *x, const int *incx);
double dnrm2_(const int *n, const double *x, const int *incx);
float cnrm2_(const int *n, const void *x, const int *incx);
double znrm2_(const int *n, const void *x, const int *incx);

/* ASUM - Sum of absolute values */
float sasum_(const int *n, const float *x, const int *incx);
double dasum_(const int *n, const double *x, const int *incx);
float casum_(const int *n, const void *x, const int *incx);
double zasum_(const int *n, const void *x, const int *incx);

/* IAMAX/IMAX - Index of maximum absolute value */
int isamax_(const int *n, const float *x, const int *incx);
int idamax_(const int *n, const double *x, const int *incx);
int icamax_(const int *n, const void *x, const int *incx);
int izamax_(const int *n, const void *x, const int *incx);

/* IAMIN/IMIN - Index of minimum absolute value */
int isamin_(const int *n, const float *x, const int *incx);
int idamin_(const int *n, const double *x, const int *incx);
int icamin_(const int *n, const void *x, const int *incx);
int izamin_(const int *n, const void *x, const int *incx);

/* SWAP - X <-> Y */
void sswap_(const int *n, float *x, const int *incx, float *y, const int *incy);
void dswap_(const int *n, double *x, const int *incx, double *y, const int *incy);
void cswap_(const int *n, void *x, const int *incx, void *y, const int *incy);
void zswap_(const int *n, void *x, const int *incx, void *y, const int *incy);

/* ROT - Givens rotation */
void srot_(const int *n, float *x, const int *incx, float *y, const int *incy, 
           const float *c, const float *s);
void drot_(const int *n, double *x, const int *incx, double *y, const int *incy, 
           const double *c, const double *s);
void crot_(const int *n, void *x, const int *incx, void *y, const int *incy, 
           const float *c, const void *s);
void zrot_(const int *n, void *x, const int *incx, void *y, const int *incy, 
           const double *c, const void *s);

/* ============================================================================
   BLAS LEVEL 2: Matrix-Vector Operations (74 operations)
   ============================================================================ */

/* GEMV - General matrix-vector multiply Y := A*X + B*Y */
void sgemv_(const char *trans, const int *m, const int *n, const float *alpha,
            const float *a, const int *lda, const float *x, const int *incx,
            const float *beta, float *y, const int *incy);
void dgemv_(const char *trans, const int *m, const int *n, const double *alpha,
            const double *a, const int *lda, const double *x, const int *incx,
            const double *beta, double *y, const int *incy);
void cgemv_(const char *trans, const int *m, const int *n, const void *alpha,
            const void *a, const int *lda, const void *x, const int *incx,
            const void *beta, void *y, const int *incy);
void zgemv_(const char *trans, const int *m, const int *n, const void *alpha,
            const void *a, const int *lda, const void *x, const int *incx,
            const void *beta, void *y, const int *incy);

/* GBMV - General banded matrix-vector multiply */
void sgbmv_(const char *trans, const int *m, const int *n, const int *kl, const int *ku,
            const float *alpha, const float *a, const int *lda, const float *x, const int *incx,
            const float *beta, float *y, const int *incy);
void dgbmv_(const char *trans, const int *m, const int *n, const int *kl, const int *ku,
            const double *alpha, const double *a, const int *lda, const double *x, const int *incx,
            const double *beta, double *y, const int *incy);
void cgbmv_(const char *trans, const int *m, const int *n, const int *kl, const int *ku,
            const void *alpha, const void *a, const int *lda, const void *x, const int *incx,
            const void *beta, void *y, const int *incy);
void zgbmv_(const char *trans, const int *m, const int *n, const int *kl, const int *ku,
            const void *alpha, const void *a, const int *lda, const void *x, const int *incx,
            const void *beta, void *y, const int *incy);

/* GER - General rank-1 update A := alpha*X*Y' + A */
void sger_(const int *m, const int *n, const float *alpha,
           const float *x, const int *incx, const float *y, const int *incy,
           float *a, const int *lda);
void dger_(const int *m, const int *n, const double *alpha,
           const double *x, const int *incx, const double *y, const int *incy,
           double *a, const int *lda);
void cgerc_(const int *m, const int *n, const void *alpha,
            const void *x, const int *incx, const void *y, const int *incy,
            void *a, const int *lda);
void cgeru_(const int *m, const int *n, const void *alpha,
            const void *x, const int *incx, const void *y, const int *incy,
            void *a, const int *lda);
void zgerc_(const int *m, const int *n, const void *alpha,
            const void *x, const int *incx, const void *y, const int *incy,
            void *a, const int *lda);
void zgeru_(const int *m, const int *n, const void *alpha,
            const void *x, const int *incx, const void *y, const int *incy,
            void *a, const int *lda);

/* HEMV - Hermitian matrix-vector multiply */
void chemv_(const char *uplo, const int *n, const void *alpha,
            const void *a, const int *lda, const void *x, const int *incx,
            const void *beta, void *y, const int *incy);
void zhemv_(const char *uplo, const int *n, const void *alpha,
            const void *a, const int *lda, const void *x, const int *incx,
            const void *beta, void *y, const int *incy);

/* HBMV - Hermitian banded matrix-vector multiply */
void chbmv_(const char *uplo, const int *n, const int *k, const void *alpha,
            const void *a, const int *lda, const void *x, const int *incx,
            const void *beta, void *y, const int *incy);
void zhbmv_(const char *uplo, const int *n, const int *k, const void *alpha,
            const void *a, const int *lda, const void *x, const int *incx,
            const void *beta, void *y, const int *incy);

/* HER - Hermitian rank-1 update */
void cher_(const char *uplo, const int *n, const float *alpha,
           const void *x, const int *incx, void *a, const int *lda);
void zher_(const char *uplo, const int *n, const double *alpha,
           const void *x, const int *incx, void *a, const int *lda);

/* HER2 - Hermitian rank-2 update */
void cher2_(const char *uplo, const int *n, const void *alpha,
            const void *x, const int *incx, const void *y, const int *incy,
            void *a, const int *lda);
void zher2_(const char *uplo, const int *n, const void *alpha,
            const void *x, const int *incx, const void *y, const int *incy,
            void *a, const int *lda);

/* SYMV - Symmetric matrix-vector multiply */
void ssymv_(const char *uplo, const int *n, const float *alpha,
            const float *a, const int *lda, const float *x, const int *incx,
            const float *beta, float *y, const int *incy);
void dsymv_(const char *uplo, const int *n, const double *alpha,
            const double *a, const int *lda, const double *x, const int *incx,
            const double *beta, double *y, const int *incy);

/* SBMV - Symmetric banded matrix-vector multiply */
void ssbmv_(const char *uplo, const int *n, const int *k, const float *alpha,
            const float *a, const int *lda, const float *x, const int *incx,
            const float *beta, float *y, const int *incy);
void dsbmv_(const char *uplo, const int *n, const int *k, const double *alpha,
            const double *a, const int *lda, const double *x, const int *incx,
            const double *beta, double *y, const int *incy);

/* SPR - Symmetric rank-1 update (packed) */
void sspr_(const char *uplo, const int *n, const float *alpha,
           const float *x, const int *incx, float *ap);
void dspr_(const char *uplo, const int *n, const double *alpha,
           const double *x, const int *incx, double *ap);

/* SPR2 - Symmetric rank-2 update (packed) */
void sspr2_(const char *uplo, const int *n, const float *alpha,
            const float *x, const int *incx, const float *y, const int *incy,
            float *ap);
void dspr2_(const char *uplo, const int *n, const double *alpha,
            const double *x, const int *incx, const double *y, const int *incy,
            double *ap);

/* SYR - Symmetric rank-1 update */
void ssyr_(const char *uplo, const int *n, const float *alpha,
           const float *x, const int *incx, float *a, const int *lda);
void dsyr_(const char *uplo, const int *n, const double *alpha,
           const double *x, const int *incx, double *a, const int *lda);

/* SYR2 - Symmetric rank-2 update */
void ssyr2_(const char *uplo, const int *n, const float *alpha,
            const float *x, const int *incx, const float *y, const int *incy,
            float *a, const int *lda);
void dsyr2_(const char *uplo, const int *n, const double *alpha,
            const double *x, const int *incx, const double *y, const int *incy,
            double *a, const int *lda);

/* TRMV - Triangular matrix-vector multiply */
void strmv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const float *a, const int *lda,
            float *x, const int *incx);
void dtrmv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const double *a, const int *lda,
            double *x, const int *incx);
void ctrmv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const void *a, const int *lda,
            void *x, const int *incx);
void ztrmv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const void *a, const int *lda,
            void *x, const int *incx);

/* TBMV - Triangular banded matrix-vector multiply */
void stbmv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const int *k, const float *a, const int *lda,
            float *x, const int *incx);
void dtbmv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const int *k, const double *a, const int *lda,
            double *x, const int *incx);
void ctbmv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const int *k, const void *a, const int *lda,
            void *x, const int *incx);
void ztbmv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const int *k, const void *a, const int *lda,
            void *x, const int *incx);

/* TRSV - Triangular matrix-vector solve */
void strsv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const float *a, const int *lda,
            float *x, const int *incx);
void dtrsv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const double *a, const int *lda,
            double *x, const int *incx);
void ctrsv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const void *a, const int *lda,
            void *x, const int *incx);
void ztrsv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const void *a, const int *lda,
            void *x, const int *incx);

/* TBSV - Triangular banded matrix-vector solve */
void stbsv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const int *k, const float *a, const int *lda,
            float *x, const int *incx);
void dtbsv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const int *k, const double *a, const int *lda,
            double *x, const int *incx);
void ctbsv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const int *k, const void *a, const int *lda,
            void *x, const int *incx);
void ztbsv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const int *k, const void *a, const int *lda,
            void *x, const int *incx);

/* TPSV - Triangular packed matrix-vector solve */
void stpsv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const float *ap, float *x, const int *incx);
void dtpsv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const double *ap, double *x, const int *incx);
void ctpsv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const void *ap, void *x, const int *incx);
void ztpsv_(const char *uplo, const char *trans, const char *diag,
            const int *n, const void *ap, void *x, const int *incx);

/* ============================================================================
   BLAS LEVEL 3: Matrix-Matrix Operations (27 operations)
   ============================================================================ */

/* GEMM - General matrix-matrix multiply C := alpha*A*B + beta*C */
void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
            const float *alpha, const float *a, const int *lda, const float *b, const int *ldb,
            const float *beta, float *c, const int *ldc);
void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
            const double *alpha, const double *a, const int *lda, const double *b, const int *ldb,
            const double *beta, double *c, const int *ldc);
void cgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
            const void *alpha, const void *a, const int *lda, const void *b, const int *ldb,
            const void *beta, void *c, const int *ldc);
void zgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
            const void *alpha, const void *a, const int *lda, const void *b, const int *ldb,
            const void *beta, void *c, const int *ldc);

/* HEMM - Hermitian matrix-matrix multiply */
void chemm_(const char *side, const char *uplo, const int *m, const int *n, const void *alpha,
            const void *a, const int *lda, const void *b, const int *ldb, const void *beta,
            void *c, const int *ldc);
void zhemm_(const char *side, const char *uplo, const int *m, const int *n, const void *alpha,
            const void *a, const int *lda, const void *b, const int *ldb, const void *beta,
            void *c, const int *ldc);

/* HER2K - Hermitian rank-2k update */
void cher2k_(const char *uplo, const char *trans, const int *n, const int *k,
             const void *alpha, const void *a, const int *lda, const void *b, const int *ldb,
             const float *beta, void *c, const int *ldc);
void zher2k_(const char *uplo, const char *trans, const int *n, const int *k,
             const void *alpha, const void *a, const int *lda, const void *b, const int *ldb,
             const double *beta, void *c, const int *ldc);

/* HERK - Hermitian rank-k update */
void cherk_(const char *uplo, const char *trans, const int *n, const int *k,
            const float *alpha, const void *a, const int *lda, const float *beta,
            void *c, const int *ldc);
void zherk_(const char *uplo, const char *trans, const int *n, const int *k,
            const double *alpha, const void *a, const int *lda, const double *beta,
            void *c, const int *ldc);

/* SYMM - Symmetric matrix-matrix multiply */
void ssymm_(const char *side, const char *uplo, const int *m, const int *n, const float *alpha,
            const float *a, const int *lda, const float *b, const int *ldb, const float *beta,
            float *c, const int *ldc);
void dsymm_(const char *side, const char *uplo, const int *m, const int *n, const double *alpha,
            const double *a, const int *lda, const double *b, const int *ldb, const double *beta,
            double *c, const int *ldc);
void csymm_(const char *side, const char *uplo, const int *m, const int *n, const void *alpha,
            const void *a, const int *lda, const void *b, const int *ldb, const void *beta,
            void *c, const int *ldc);
void zsymm_(const char *side, const char *uplo, const int *m, const int *n, const void *alpha,
            const void *a, const int *lda, const void *b, const int *ldb, const void *beta,
            void *c, const int *ldc);

/* SYR2K - Symmetric rank-2k update */
void ssyr2k_(const char *uplo, const char *trans, const int *n, const int *k, const float *alpha,
             const float *a, const int *lda, const float *b, const int *ldb, const float *beta,
             float *c, const int *ldc);
void dsyr2k_(const char *uplo, const char *trans, const int *n, const int *k, const double *alpha,
             const double *a, const int *lda, const double *b, const int *ldb, const double *beta,
             double *c, const int *ldc);
void csyr2k_(const char *uplo, const char *trans, const int *n, const int *k, const void *alpha,
             const void *a, const int *lda, const void *b, const int *ldb, const void *beta,
             void *c, const int *ldc);
void zsyr2k_(const char *uplo, const char *trans, const int *n, const int *k, const void *alpha,
             const void *a, const int *lda, const void *b, const int *ldb, const void *beta,
             void *c, const int *ldc);

/* SYRK - Symmetric rank-k update */
void ssyrk_(const char *uplo, const char *trans, const int *n, const int *k, const float *alpha,
            const float *a, const int *lda, const float *beta, float *c, const int *ldc);
void dsyrk_(const char *uplo, const char *trans, const int *n, const int *k, const double *alpha,
            const double *a, const int *lda, const double *beta, double *c, const int *ldc);
void csyrk_(const char *uplo, const char *trans, const int *n, const int *k, const void *alpha,
            const void *a, const int *lda, const void *beta, void *c, const int *ldc);
void zsyrk_(const char *uplo, const char *trans, const int *n, const int *k, const void *alpha,
            const void *a, const int *lda, const void *beta, void *c, const int *ldc);

/* TRMM - Triangular matrix-matrix multiply */
void strmm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const float *alpha, const float *a, const int *lda,
            float *b, const int *ldb);
void dtrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const double *alpha, const double *a, const int *lda,
            double *b, const int *ldb);
void ctrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const void *alpha, const void *a, const int *lda,
            void *b, const int *ldb);
void ztrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const void *alpha, const void *a, const int *lda,
            void *b, const int *ldb);

/* TRSM - Triangular matrix-matrix solve */
void strsm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const float *alpha, const float *a, const int *lda,
            float *b, const int *ldb);
void dtrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const double *alpha, const double *a, const int *lda,
            double *b, const int *ldb);
void ctrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const void *alpha, const void *a, const int *lda,
            void *b, const int *ldb);
void ztrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const void *alpha, const void *a, const int *lda,
            void *b, const int *ldb);

/* ============================================================================
   LAPACK Auxiliary Functions (~400 operations)
   ============================================================================ */

/* LA routines - Utility functions for LAPACK */
/* LASCL - Scale a matrix */
void slascl_(const char *type, const int *kl, const int *ku, const float *cfrom, const float *cto,
             const int *m, const int *n, float *a, const int *lda, int *info);
void dlascl_(const char *type, const int *kl, const int *ku, const double *cfrom, const double *cto,
             const int *m, const int *n, double *a, const int *lda, int *info);
void clascl_(const char *type, const int *kl, const int *ku, const float *cfrom, const float *cto,
             const int *m, const int *n, void *a, const int *lda, int *info);
void zlascl_(const char *type, const int *kl, const int *ku, const double *cfrom, const double *cto,
             const int *m, const int *n, void *a, const int *lda, int *info);

/* LARF - Apply Householder reflection */
void slarf_(const char *side, const int *m, const int *n, const float *v, const int *incv,
            const float *tau, float *c, const int *ldc, float *work);
void dlarf_(const char *side, const int *m, const int *n, const double *v, const int *incv,
            const double *tau, double *c, const int *ldc, double *work);
void clarf_(const char *side, const int *m, const int *n, const void *v, const int *incv,
            const void *tau, void *c, const int *ldc, void *work);
void zlarf_(const char *side, const int *m, const int *n, const void *v, const int *incv,
            const void *tau, void *c, const int *ldc, void *work);

/* LARFG - Generate Householder reflection */
void slarfg_(const int *n, float *alpha, float *x, const int *incx, float *tau);
void dlarfg_(const int *n, double *alpha, double *x, const int *incx, double *tau);
void clarfg_(const int *n, void *alpha, void *x, const int *incx, void *tau);
void zlarfg_(const int *n, void *alpha, void *x, const int *incx, void *tau);

/* GEBAL - Balance a general matrix */
void sgebal_(const char *job, const int *n, float *a, const int *lda, int *ilo, int *ihi,
             float *scale, int *info);
void dgebal_(const char *job, const int *n, double *a, const int *lda, int *ilo, int *ihi,
             double *scale, int *info);
void cgebal_(const char *job, const int *n, void *a, const int *lda, int *ilo, int *ihi,
             float *scale, int *info);
void zgebal_(const char *job, const int *n, void *a, const int *lda, int *ilo, int *ihi,
             double *scale, int *info);

/* GEHRD - Reduce to Hessenberg form */
void sgehrd_(const int *n, const int *ilo, const int *ihi, float *a, const int *lda,
             float *tau, float *work, const int *lwork, int *info);
void dgehrd_(const int *n, const int *ilo, const int *ihi, double *a, const int *lda,
             double *tau, double *work, const int *lwork, int *info);
void cgehrd_(const int *n, const int *ilo, const int *ihi, void *a, const int *lda,
             void *tau, void *work, const int *lwork, int *info);
void zgehrd_(const int *n, const int *ilo, const int *ihi, void *a, const int *lda,
             void *tau, void *work, const int *lwork, int *info);

/* ============================================================================
   LAPACK Computational Functions (~800 operations)
   ============================================================================ */

/* GETRF - LU factorization */
void sgetrf_(const int *m, const int *n, float *a, const int *lda, int *ipiv, int *info);
void dgetrf_(const int *m, const int *n, double *a, const int *lda, int *ipiv, int *info);
void cgetrf_(const int *m, const int *n, void *a, const int *lda, int *ipiv, int *info);
void zgetrf_(const int *m, const int *n, void *a, const int *lda, int *ipiv, int *info);

/* GETRS - Solve after LU factorization */
void sgetrs_(const char *trans, const int *n, const int *nrhs, const float *a, const int *lda,
             const int *ipiv, float *b, const int *ldb, int *info);
void dgetrs_(const char *trans, const int *n, const int *nrhs, const double *a, const int *lda,
             const int *ipiv, double *b, const int *ldb, int *info);
void cgetrs_(const char *trans, const int *n, const int *nrhs, const void *a, const int *lda,
             const int *ipiv, void *b, const int *ldb, int *info);
void zgetrs_(const char *trans, const int *n, const int *nrhs, const void *a, const int *lda,
             const int *ipiv, void *b, const int *ldb, int *info);

/* GEQRF - QR factorization */
void sgeqrf_(const int *m, const int *n, float *a, const int *lda, float *tau,
             float *work, const int *lwork, int *info);
void dgeqrf_(const int *m, const int *n, double *a, const int *lda, double *tau,
             double *work, const int *lwork, int *info);
void cgeqrf_(const int *m, const int *n, void *a, const int *lda, void *tau,
             void *work, const int *lwork, int *info);
void zgeqrf_(const int *m, const int *n, void *a, const int *lda, void *tau,
             void *work, const int *lwork, int *info);

/* POTRF - Cholesky factorization */
void spotrf_(const char *uplo, const int *n, float *a, const int *lda, int *info);
void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);
void cpotrf_(const char *uplo, const int *n, void *a, const int *lda, int *info);
void zpotrf_(const char *uplo, const int *n, void *a, const int *lda, int *info);

/* POTRS - Solve after Cholesky factorization */
void spotrs_(const char *uplo, const int *n, const int *nrhs, const float *a, const int *lda,
             float *b, const int *ldb, int *info);
void dpotrs_(const char *uplo, const int *n, const int *nrhs, const double *a, const int *lda,
             double *b, const int *ldb, int *info);
void cpotrs_(const char *uplo, const int *n, const int *nrhs, const void *a, const int *lda,
             void *b, const int *ldb, int *info);
void zpotrs_(const char *uplo, const int *n, const int *nrhs, const void *a, const int *lda,
             void *b, const int *ldb, int *info);

/* GESVD - Singular value decomposition */
void sgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n, float *a,
             const int *lda, float *s, float *u, const int *ldu, float *vt, const int *ldvt,
             float *work, const int *lwork, int *info);
void dgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n, double *a,
             const int *lda, double *s, double *u, const int *ldu, double *vt, const int *ldvt,
             double *work, const int *lwork, int *info);
void cgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n, void *a,
             const int *lda, float *s, void *u, const int *ldu, void *vt, const int *ldvt,
             void *work, const int *lwork, int *info);
void zgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n, void *a,
             const int *lda, double *s, void *u, const int *ldu, void *vt, const int *ldvt,
             void *work, const int *lwork, int *info);

/* GESDD - Divide-and-conquer SVD */
void sgesdd_(const char *jobz, const int *m, const int *n, float *a, const int *lda,
             float *s, float *u, const int *ldu, float *vt, const int *ldvt,
             float *work, const int *lwork, int *iwork, int *info);
void dgesdd_(const char *jobz, const int *m, const int *n, double *a, const int *lda,
             double *s, double *u, const int *ldu, double *vt, const int *ldvt,
             double *work, const int *lwork, int *iwork, int *info);
void cgesdd_(const char *jobz, const int *m, const int *n, void *a, const int *lda,
             float *s, void *u, const int *ldu, void *vt, const int *ldvt,
             void *work, const int *lwork, int *iwork, int *info);
void zgesdd_(const char *jobz, const int *m, const int *n, void *a, const int *lda,
             double *s, void *u, const int *ldu, void *vt, const int *ldvt,
             void *work, const int *lwork, int *iwork, int *info);

/* SYTRF - Bunch-Kaufman factorization */
void ssytrf_(const char *uplo, const int *n, float *a, const int *lda, int *ipiv,
             float *work, const int *lwork, int *info);
void dsytrf_(const char *uplo, const int *n, double *a, const int *lda, int *ipiv,
             double *work, const int *lwork, int *info);
void chetrf_(const char *uplo, const int *n, void *a, const int *lda, int *ipiv,
             void *work, const int *lwork, int *info);
void zhetrf_(const char *uplo, const int *n, void *a, const int *lda, int *ipiv,
             void *work, const int *lwork, int *info);

/* STEQR - Tridiagonal eigenvalue problem via QR */
void ssteqr_(const char *compz, const int *n, float *d, float *e, float *z, const int *ldz,
             float *work, int *info);
void dsteqr_(const char *compz, const int *n, double *d, double *e, double *z, const int *ldz,
             double *work, int *info);
void csteqr_(const char *compz, const int *n, float *d, float *e, void *z, const int *ldz,
             float *work, int *info);
void zsteqr_(const char *compz, const int *n, double *d, double *e, void *z, const int *ldz,
             double *work, int *info);

/* ============================================================================
   LAPACK Driver Functions (~20 operations)
   ============================================================================ */

/* GESV - Solve general linear system */
void sgesv_(const int *n, const int *nrhs, float *a, const int *lda, int *ipiv,
            float *b, const int *ldb, int *info);
void dgesv_(const int *n, const int *nrhs, double *a, const int *lda, int *ipiv,
            double *b, const int *ldb, int *info);
void cgesv_(const int *n, const int *nrhs, void *a, const int *lda, int *ipiv,
            void *b, const int *ldb, int *info);
void zgesv_(const int *n, const int *nrhs, void *a, const int *lda, int *ipiv,
            void *b, const int *ldb, int *info);

/* POSV - Solve symmetric positive definite system */
void sposv_(const char *uplo, const int *n, const int *nrhs, float *a, const int *lda,
            float *b, const int *ldb, int *info);
void dposv_(const char *uplo, const int *n, const int *nrhs, double *a, const int *lda,
            double *b, const int *ldb, int *info);
void cposv_(const char *uplo, const int *n, const int *nrhs, void *a, const int *lda,
            void *b, const int *ldb, int *info);
void zposv_(const char *uplo, const int *n, const int *nrhs, void *a, const int *lda,
            void *b, const int *ldb, int *info);

/* GEEV - Eigenvalue decomposition */
void sgeev_(const char *jobvl, const char *jobvr, const int *n, float *a, const int *lda,
            float *wr, float *wi, float *vl, const int *ldvl, float *vr, const int *ldvr,
            float *work, const int *lwork, int *info);
void dgeev_(const char *jobvl, const char *jobvr, const int *n, double *a, const int *lda,
            double *wr, double *wi, double *vl, const int *ldvl, double *vr, const int *ldvr,
            double *work, const int *lwork, int *info);
void cgeev_(const char *jobvl, const char *jobvr, const int *n, void *a, const int *lda,
            void *w, void *vl, const int *ldvl, void *vr, const int *ldvr,
            void *work, const int *lwork, float *rwork, int *info);
void zgeev_(const char *jobvl, const char *jobvr, const int *n, void *a, const int *lda,
            void *w, void *vl, const int *ldvl, void *vr, const int *ldvr,
            void *work, const int *lwork, double *rwork, int *info);

/* SYEV - Symmetric eigenvalue decomposition */
void ssyev_(const char *jobz, const char *uplo, const int *n, float *a, const int *lda,
            float *w, float *work, const int *lwork, int *info);
void dsyev_(const char *jobz, const char *uplo, const int *n, double *a, const int *lda,
            double *w, double *work, const int *lwork, int *info);
void cheev_(const char *jobz, const char *uplo, const int *n, void *a, const int *lda,
            float *w, void *work, const int *lwork, float *rwork, int *info);
void zheev_(const char *jobz, const char *uplo, const int *n, void *a, const int *lda,
            double *w, void *work, const int *lwork, double *rwork, int *info);

#ifdef __cplusplus
}
#endif

#endif /* BLAS_LAPACK_REFERENCE_COMPLETE_H */
