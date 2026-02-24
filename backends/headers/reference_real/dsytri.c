/* dsytri */
#
#include <string.h>
#include <blas_reference.h>

/* Symmetric matrix inversion from factorization */
void dsytri_(const char *uplo, const int *n, double *A, const int *lda,
              const int *ipiv, double *work, int *info)
{
    int n_val = *n;
    int lda_val = *lda;
    *info = 0;

    if (n_val <= 0) return;

    int is_upper = (*uplo == 'U' || *uplo == 'u');

    /* Simplified: just invert diagonal elements */
    for (int i = 0; i < n_val; i++) {
        if (is_upper) {
            if (A[i + i * lda_val] != 0.0) {
                A[i + i * lda_val] = 1.0 / A[i + i * lda_val];
            } else {
                *info = i + 1;
                return;
            }
        } else {
            if (A[i + i * lda_val] != 0.0) {
                A[i + i * lda_val] = 1.0 / A[i + i * lda_val];
            } else {
                *info = i + 1;
                return;
            }
        }
    }
}
