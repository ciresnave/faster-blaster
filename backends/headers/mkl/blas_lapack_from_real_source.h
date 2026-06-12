#ifndef BLAS_LAPACK_FROM_REAL_SOURCE_H
#define BLAS_LAPACK_FROM_REAL_SOURCE_H

/* ============================================================================
   ACTUAL BLAS/LAPACK Wrapper Declarations
   Extracted directly from blas-lapack-reference/src/blas_wrappers.c
   These are real, production wrapper functions that exist in the codebase
   ============================================================================ */

/* Level 1: Vector operations */
float sdot_(const int *n, const float *x, const int *incx, 
            const float *y, const int *incy);
double ddot_(const int *n, const double *x, const int *incx, 
             const double *y, const int *incy);
void saxpy_(const int *n, const float *alpha, const float *x, const int *incx,
            float *y, const int *incy);
void daxpy_(const int *n, const double *alpha, const double *x, const int *incx,
            double *y, const int *incy);
void scopy_(const int *n, const float *x, const int *incx, float *y, const int *incy);
void dcopy_(const int *n, const double *x, const int *incx, double *y, const int *incy);
void sscal_(const int *n, const float *alpha, float *x, const int *incx);
void dscal_(const int *n, const double *alpha, double *x, const int *incx);
float snrm2_(const int *n, const float *x, const int *incx);
double dnrm2_(const int *n, const double *x, const int *incx);
float sasum_(const int *n, const float *x, const int *incx);
double dasum_(const int *n, const double *x, const int *incx);
int isamax_(const int *n, const float *x, const int *incx);
int idamax_(const int *n, const double *x, const int *incx);
void sswap_(const int *n, float *x, const int *incx, float *y, const int *incy);
void dswap_(const int *n, double *x, const int *incx, double *y, const int *incy);

/* Level 2: Matrix-vector operations */
void sgemv_(const char *trans, const int *m, const int *n, const float *alpha,
            const float *A, const int *lda, const float *x, const int *incx,
            const float *beta, float *y, const int *incy);
void dgemv_(const char *trans, const int *m, const int *n, const double *alpha,
            const double *A, const int *lda, const double *x, const int *incx,
            const double *beta, double *y, const int *incy);
void sger_(const int *m, const int *n, const float *alpha,
           const float *x, const int *incx, const float *y, const int *incy,
           float *A, const int *lda);
void dger_(const int *m, const int *n, const double *alpha,
           const double *x, const int *incx, const double *y, const int *incy,
           double *A, const int *lda);

/* Level 3: Matrix-matrix operations */
void sgemm_(const char *transa, const char *transb, const int *m, const int *n, 
            const int *k, const float *alpha, const float *A, const int *lda, 
            const float *B, const int *ldb, const float *beta, float *C, const int *ldc);
void dgemm_(const char *transa, const char *transb, const int *m, const int *n, 
            const int *k, const double *alpha, const double *A, const int *lda, 
            const double *B, const int *ldb, const double *beta, double *C, const int *ldc);
void strmm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const float *alpha, const float *A, const int *lda,
            float *B, const int *ldb);
void dtrmm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const double *alpha, const double *A, const int *lda,
            double *B, const int *ldb);
void strsm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const float *alpha, const float *A, const int *lda,
            float *B, const int *ldb);
void dtrsm_(const char *side, const char *uplo, const char *transa, const char *diag,
            const int *m, const int *n, const double *alpha, const double *A, const int *lda,
            double *B, const int *ldb);

/* LAPACK wrappers */
void ssteqr_(const char *compz, const int *n, float *d, float *e, float *Z, 
             const int *ldz, float *work, int *info);
void dsteqr_(const char *compz, const int *n, double *d, double *e, double *Z, 
             const int *ldz, double *work, int *info);
void steqr_(const char *compz, const int *n, float *d, float *e, float *Z, 
            const int *ldz, float *work, int *info);
void dsteqr_wrapper_(const char *compz, const int *n, double *d, double *e, double *Z,
                     const int *ldz, double *work, int *info);

#endif
