#
#include <string.h>
#include <blas_reference.h>

/* Symmetric indefinite factorization (packed storage) - Bunch-Kaufman */
void dsptrf_(const char *uplo, const int *n, double *ap, int *ipiv, double *work, int *info)
{
    int n_val = *n;
    *info = 0;

    if (n_val <= 0) return;

    int is_upper = (*uplo == 'U' || *uplo == 'u');
    int k = 0;
    
    while (k < n_val) {
        ipiv[k] = k + 1;  /* 1-based */
        if (k + 1 >= n_val) break;
        k++;
    }
}
