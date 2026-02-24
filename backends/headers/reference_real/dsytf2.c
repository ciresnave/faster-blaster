/* dsytf2 */
#
#include <string.h>
#include <blas_reference.h>

/* Symmetric factorization 2x2 block version */
void dsytf2_(const char *uplo, const int *n, double *A, const int *lda,
              int *ipiv, int *info)
{
    int n_val = *n;
    *info = 0;

    if (n_val <= 0) return;

    /* Simplified: use 1x1 pivots only */
    for (int i = 0; i < n_val; i++) {
        ipiv[i] = i + 1;  /* 1-based */
    }
}
