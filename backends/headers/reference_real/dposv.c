/* Forward declarations of Fortran-style functions */
void dpotrf_(const char *uplo, const int *n, double *A, const int *lda, int *info);
void dpotrs_(const char *uplo, const int *n, const int *nrhs,
             const double *A, const int *lda, double *B, const int *ldb, int *info);

void dposv_(const char *uplo, const int *n, const int *nrhs, double *A, const int *lda,
            double *B, const int *ldb, int *info)
{
    if (*n <= 0 || *nrhs <= 0) {
        *info = 0;
        return;
    }
    
    /* Compute Cholesky factorization */
    dpotrf_(uplo, n, A, lda, info);
    
    if (*info == 0) {
        /* Solve using Cholesky factorization */
        dpotrs_(uplo, n, nrhs, A, lda, B, ldb, info);
    }
}
