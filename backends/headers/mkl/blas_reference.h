/**
 * Sample BLAS Reference Header for Testing
 * This demonstrates the format that fb_codegen expects
 */

#ifndef BLAS_REFERENCE_H
#define BLAS_REFERENCE_H

/* BLAS Level 1 Operations */

void cblas_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy);
void cblas_daxpy(int n, double alpha, const double* x, int incx, double* y, int incy);
void cblas_caxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy);
void cblas_zaxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy);

float cblas_sdot(int n, const float* x, int incx, const float* y, int incy);
double cblas_ddot(int n, const double* x, int incx, const double* y, int incy);

void cblas_scopy(int n, const float* x, int incx, float* y, int incy);
void cblas_dcopy(int n, const double* x, int incx, double* y, int incy);

void cblas_scal(int n, float alpha, float* x, int incx);
void cblas_dscal(int n, double alpha, double* x, int incx);

float cblas_snrm2(int n, const float* x, int incx);
double cblas_dnrm2(int n, const double* x, int incx);

float cblas_sasum(int n, const float* x, int incx);
double cblas_dasum(int n, const double* x, int incx);

int cblas_isamax(int n, const float* x, int incx);
int cblas_idamax(int n, const double* x, int incx);

/* BLAS Level 2 Operations */

void cblas_sgemv(int order, int trans, int m, int n, float alpha,
                 const float* a, int lda, const float* x, int incx,
                 float beta, float* y, int incy);
void cblas_dgemv(int order, int trans, int m, int n, double alpha,
                 const double* a, int lda, const double* x, int incx,
                 double beta, double* y, int incy);

void cblas_sger(int order, int m, int n, float alpha,
                const float* x, int incx, const float* y, int incy,
                float* a, int lda);
void cblas_dger(int order, int m, int n, double alpha,
                const double* x, int incx, const double* y, int incy,
                double* a, int lda);

void cblas_strmv(int order, int uplo, int trans, int diag,
                 int n, const float* a, int lda, float* x, int incx);
void cblas_dtrmv(int order, int uplo, int trans, int diag,
                 int n, const double* a, int lda, double* x, int incx);

/* BLAS Level 3 Operations */

void cblas_sgemm(int order, int transa, int transb, int m, int n, int k,
                 float alpha, const float* a, int lda, const float* b, int ldb,
                 float beta, float* c, int ldc);
void cblas_dgemm(int order, int transa, int transb, int m, int n, int k,
                 double alpha, const double* a, int lda, const double* b, int ldb,
                 double beta, double* c, int ldc);

void cblas_ssymm(int order, int side, int uplo, int m, int n,
                 float alpha, const float* a, int lda, const float* b, int ldb,
                 float beta, float* c, int ldc);
void cblas_dsymm(int order, int side, int uplo, int m, int n,
                 double alpha, const double* a, int lda, const double* b, int ldb,
                 double beta, double* c, int ldc);

void cblas_strmm(int order, int side, int uplo, int trans, int diag,
                 int m, int n, float alpha, const float* a, int lda,
                 float* b, int ldb);
void cblas_dtrmm(int order, int side, int uplo, int trans, int diag,
                 int m, int n, double alpha, const double* a, int lda,
                 double* b, int ldb);

void cblas_strsm(int order, int side, int uplo, int trans, int diag,
                 int m, int n, float alpha, const float* a, int lda,
                 float* b, int ldb);
void cblas_dtrsm(int order, int side, int uplo, int trans, int diag,
                 int m, int n, double alpha, const double* a, int lda,
                 double* b, int ldb);

#endif /* BLAS_REFERENCE_H */
