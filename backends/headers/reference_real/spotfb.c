/* spotfb */
#
#include <string.h>
#include <blas_reference.h>

/* Blocked Cholesky factorization */
void spotfb_(const char *uplo, const int *n, const int *nb, float *A, const int *lda, int *info)
{
    int n_val = *n;
    int nb_val = *nb;
    int lda_val = *lda;
    *info = 0;

    if (n_val <= 0 || nb_val <= 0) return;

    /* Use SPOTRF for blocks */
    for (int i = 0; i < n_val; i += nb_val) {
        int ib = (i + nb_val < n_val) ? nb_val : (n_val - i);
        
        /* Factor block */
        spotrf_(uplo, &ib, &A[i + i * lda_val], &lda_val, info);
        if (*info != 0) return;
    }
}
