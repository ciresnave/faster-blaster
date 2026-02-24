/* Forward declarations of Fortran-style functions */
void spotrf_(const char *uplo, const int *n, float *A, const int *lda, int *info);
void spotrs_(const char *uplo, const int *n, const int *nrhs,
             const float *A, const int *lda, float *B, const int *ldb, int *info);

void sposv_(const char *uplo, const int *n, const int *nrhs, float *A, const int *lda,
            float *B, const int *ldb, int *info)
{
    if (*n <= 0 || *nrhs <= 0) {
        *info = 0;
        return;
    }
    
    /* Compute Cholesky factorization */
    spotrf_(uplo, n, A, lda, info);
    
    if (*info == 0) {
        /* Solve using Cholesky factorization */
        spotrs_(uplo, n, nrhs, A, lda, B, ldb, info);
    }
}
