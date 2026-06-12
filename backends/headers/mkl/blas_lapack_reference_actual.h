#ifndef BLAS_LAPACK_REFERENCE_ACTUAL_H
#define BLAS_LAPACK_REFERENCE_ACTUAL_H

/* ============================================================================
   ACTUAL BLAS/LAPACK Reference Implementation Function Declarations
   Auto-extracted from blas-lapack-reference source
   ============================================================================ */

/* BLAS Level 1 */
float sdot_(const int *n, const float *x, const int *incx, const float *y, const int *incy);
double ddot_(const int *n, const double *x, const int *incx, const double *y, const int *incy);
void saxpy_(const int *n, const float *alpha, const float *x, const int *incx, float *y, const int *incy);
void daxpy_(const int *n, const double *alpha, const double *x, const int *incx, double *y, const int *incy);
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

/* BLAS Level 2 */
void sgemv_(const char *trans, const int *m, const int *n, const float *alpha, const float *a, const int *lda, const float *x, const int *incx, const float *beta, float *y, const int *incy);
void dgemv_(const char *trans, const int *m, const int *n, const double *alpha, const double *a, const int *lda, const double *x, const int *incx, const double *beta, double *y, const int *incy);
void sger_(const int *m, const int *n, const float *alpha, const float *x, const int *incx, const float *y, const int *incy, float *a, const int *lda);
void dger_(const int *m, const int *n, const double *alpha, const double *x, const int *incx, const double *y, const int *incy, double *a, const int *lda);
void strmv_(const char *uplo, const char *trans, const char *diag, const int *n, const float *a, const int *lda, float *x, const int *incx);
void dtrmv_(const char *uplo, const char *trans, const char *diag, const int *n, const double *a, const int *lda, double *x, const int *incx);
void strsv_(const char *uplo, const char *trans, const char *diag, const int *n, const float *a, const int *lda, float *x, const int *incx);
void dtrsv_(const char *uplo, const char *trans, const char *diag, const int *n, const double *a, const int *lda, double *x, const int *incx);

/* BLAS Level 3 */
void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, const float *alpha, const float *a, const int *lda, const float *b, const int *ldb, const float *beta, float *c, const int *ldc);
void dgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k, const double *alpha, const double *a, const int *lda, const double *b, const int *ldb, const double *beta, double *c, const int *ldc);
void strmm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, const float *alpha, const float *a, const int *lda, float *b, const int *ldb);
void dtrmm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, const double *alpha, const double *a, const int *lda, double *b, const int *ldb);
void strsm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, const float *alpha, const float *a, const int *lda, float *b, const int *ldb);
void dtrsm_(const char *side, const char *uplo, const char *transa, const char *diag, const int *m, const int *n, const double *alpha, const double *a, const int *lda, double *b, const int *ldb);

/* LAPACK Auxiliary */
void slascl_(const char *type, const int *kl, const int *ku, const float *cfrom, const float *cto, const int *m, const int *n, float *a, const int *lda, int *info);
void dlascl_(const char *type, const int *kl, const int *ku, const double *cfrom, const double *cto, const int *m, const int *n, double *a, const int *lda, int *info);
void slarf_(const char *side, const int *m, const int *n, const float *v, const int *incv, const float *tau, float *c, const int *ldc, float *work);
void dlarf_(const char *side, const int *m, const int *n, const double *v, const int *incv, const double *tau, double *c, const int *ldc, double *work);
void slarfg_(const int *n, float *alpha, float *x, const int *incx, float *tau);
void dlarfg_(const int *n, double *alpha, double *x, const int *incx, double *tau);
void sgebal_(const char *job, const int *n, float *a, const int *lda, int *ilo, int *ihi, float *scale, int *info);
void dgebal_(const char *job, const int *n, double *a, const int *lda, int *ilo, int *ihi, double *scale, int *info);
void sgehrd_(const int *n, const int *ilo, const int *ihi, float *a, const int *lda, float *tau, float *work, const int *lwork, int *info);
void dgehrd_(const int *n, const int *ilo, const int *ihi, double *a, const int *lda, double *tau, double *work, const int *lwork, int *info);

/* LAPACK Computational */
void sgetrf_(const int *m, const int *n, float *a, const int *lda, int *ipiv, int *info);
void dgetrf_(const int *m, const int *n, double *a, const int *lda, int *ipiv, int *info);
void sgetrs_(const char *trans, const int *n, const int *nrhs, const float *a, const int *lda, const int *ipiv, float *b, const int *ldb, int *info);
void dgetrs_(const char *trans, const int *n, const int *nrhs, const double *a, const int *lda, const int *ipiv, double *b, const int *ldb, int *info);
void sgeqrf_(const int *m, const int *n, float *a, const int *lda, float *tau, float *work, const int *lwork, int *info);
void dgeqrf_(const int *m, const int *n, double *a, const int *lda, double *tau, double *work, const int *lwork, int *info);
void spotrf_(const char *uplo, const int *n, float *a, const int *lda, int *info);
void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);
void sgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n, float *a, const int *lda, float *s, float *u, const int *ldu, float *vt, const int *ldvt, float *work, const int *lwork, int *info);
void dgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n, double *a, const int *lda, double *s, double *u, const int *ldu, double *vt, const int *ldvt, double *work, const int *lwork, int *info);
void ssteqr_(const char *compz, const int *n, float *d, float *e, float *z, const int *ldz, float *work, int *info);
void dsteqr_(const char *compz, const int *n, double *d, double *e, double *z, const int *ldz, double *work, int *info);

/* LAPACK Drivers */
void sgesv_(const int *n, const int *nrhs, float *a, const int *lda, int *ipiv, float *b, const int *ldb, int *info);
void dgesv_(const int *n, const int *nrhs, double *a, const int *lda, int *ipiv, double *b, const int *ldb, int *info);
void sposv_(const char *uplo, const int *n, const int *nrhs, float *a, const int *lda, float *b, const int *ldb, int *info);
void dposv_(const char *uplo, const int *n, const int *nrhs, double *a, const int *lda, double *b, const int *ldb, int *info);

#endif
