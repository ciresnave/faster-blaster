/* ssytri */
#
#include <string.h>
#include <blas_reference.h>

/* Symmetric matrix inversion from factorization */
void ssytri_(const char *uplo, const int *n, float *A, const int *lda,
              const int *ipiv, float *work, int *info)
{
    int n_val = *n;
    int lda_val = *lda;
    *info = 0;

    if (n_val <= 0) return;

    int is_upper = (*uplo == 'U' || *uplo == 'u');

    /* Simplified: just invert diagonal elements */
    for (int i = 0; i < n_val; i++) {
        if (is_upper) {
            if (A[i + i * lda_val] != 0.0f) {
                A[i + i * lda_val] = 1.0f / A[i + i * lda_val];
            } else {
                *info = i + 1;
                return;
            }
        } else {
            if (A[i + i * lda_val] != 0.0f) {
                A[i + i * lda_val] = 1.0f / A[i + i * lda_val];
            } else {
                *info = i + 1;
                return;
            }
        }
    }
}
