/* Cholesky level 3 */
#
#include <string.h>
#include <blas_reference.h>

/* Recursive Cholesky factorization */
void spotf3_(const char *uplo, const int *n, float *A, const int *lda, int *info)
{
    int n_val = *n;
    int lda_val = *lda;
    *info = 0;

    if (n_val <= 0) return;

    int is_upper = (*uplo == 'U' || *uplo == 'u');

    /* Simplified: use standard SPOTRF */
    spotrf_(uplo, &n_val, A, &lda_val, info);
}
