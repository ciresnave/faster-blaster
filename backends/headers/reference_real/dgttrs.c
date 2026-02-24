#
#include <string.h>
#include <blas_reference.h>

/* Solve tridiagonal system using LU factorization from DGTTRF */
void dgttrs_(const char *trans, const int *n, const int *nrhs, double *dl,
              double *d, double *du, double *du2, const int *ipiv,
              double *B, const int *ldb, int *info)
{
    int n_val = *n;
    int nrhs_val = *nrhs;
    int ldb_val = *ldb;
    *info = 0;

    if (n_val <= 0 || nrhs_val <= 0) return;

    /* Apply permutations from factorization */
    for (int i = 0; i < n_val; i++) {
        int piv_idx = ipiv[i] - 1;  /* Convert to 0-based */
        if (piv_idx != i) {
            for (int j = 0; j < nrhs_val; j++) {
                double tmp = B[i + j * ldb_val];
                B[i + j * ldb_val] = B[piv_idx + j * ldb_val];
                B[piv_idx + j * ldb_val] = tmp;
            }
        }
    }

    /* Forward substitution (L*y = permuted B) */
    for (int i = 0; i < n_val - 1; i++) {
        for (int j = 0; j < nrhs_val; j++) {
            B[i + 1 + j * ldb_val] -= dl[i] * B[i + j * ldb_val];
        }
    }

    /* Diagonal scaling (D*z = y) and back substitution (U*x = z) */
    for (int j = 0; j < nrhs_val; j++) {
        B[n_val - 1 + j * ldb_val] /= d[n_val - 1];
        if (n_val > 1) {
            B[n_val - 2 + j * ldb_val] = (B[n_val - 2 + j * ldb_val] - du[n_val - 2] * B[n_val - 1 + j * ldb_val]) / d[n_val - 2];
        }
        for (int i = n_val - 3; i >= 0; i--) {
            B[i + j * ldb_val] = (B[i + j * ldb_val] - du[i] * B[i + 1 + j * ldb_val] - du2[i] * B[i + 2 + j * ldb_val]) / d[i];
        }
    }
}
