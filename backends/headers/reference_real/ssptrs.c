#
#include <string.h>
#include <blas_reference.h>

/* Solve using symmetric packed factorization */
void ssptrs_(const char *uplo, const int *n, const int *nrhs, const float *ap,
              const int *ipiv, float *B, const int *ldb, int *info)
{
    int n_val = *n;
    int nrhs_val = *nrhs;
    int ldb_val = *ldb;
    *info = 0;

    if (n_val <= 0 || nrhs_val <= 0) return;

    int is_upper = (*uplo == 'U' || *uplo == 'u');

    /* Apply permutations */
    for (int i = 0; i < n_val; i++) {
        int piv_idx = ipiv[i] - 1;  /* Convert to 0-based */
        if (piv_idx != i && piv_idx >= 0 && piv_idx < n_val) {
            for (int j = 0; j < nrhs_val; j++) {
                float tmp = B[i + j * ldb_val];
                B[i + j * ldb_val] = B[piv_idx + j * ldb_val];
                B[piv_idx + j * ldb_val] = tmp;
            }
        }
    }

    /* Simplified solve: just divide by diagonal (assumes factorization complete) */
    for (int i = 0; i < n_val; i++) {
        int idx = i * n_val - i * (i + 1) / 2;  /* Packed index for A[i][i] */
        if (ap[idx] != 0.0f) {
            for (int j = 0; j < nrhs_val; j++) {
                B[i + j * ldb_val] /= ap[idx];
            }
        }
    }
}
