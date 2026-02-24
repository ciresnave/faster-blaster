#
#include <string.h>
#include <blas_reference.h>

/* Solve tridiagonal system T*X = B using Gaussian elimination */
void sgtsv_(const int *n, const int *nrhs, float *dl, float *d, float *du,
             float *B, const int *ldb, int *info)
{
    int n_val = *n;
    int nrhs_val = *nrhs;
    int ldb_val = *ldb;
    *info = 0;

    if (n_val <= 0 || nrhs_val <= 0) return;

    /* Check for bad diagonal in first row */
    if (d[0] == 0.0f) {
        *info = 1;
        return;
    }

    /* Forward elimination: transform to upper triangular */
    for (int i = 0; i < n_val - 1; i++) {
        /* Scale first subdiagonal element */
        dl[i] /= d[i];

        /* Update diagonal element */
        if (i + 1 < n_val) {
            d[i + 1] -= dl[i] * du[i];
            if (d[i + 1] == 0.0f) {
                *info = i + 2;
                return;
            }
        }

        /* Update right-hand side vectors */
        for (int j = 0; j < nrhs_val; j++) {
            B[(i + 1) + j * ldb_val] -= dl[i] * B[i + j * ldb_val];
        }
    }

    /* Back substitution */
    for (int j = 0; j < nrhs_val; j++) {
        B[n_val - 1 + j * ldb_val] /= d[n_val - 1];
        for (int i = n_val - 2; i >= 0; i--) {
            B[i + j * ldb_val] = (B[i + j * ldb_val] - du[i] * B[i + 1 + j * ldb_val]) / d[i];
        }
    }
}
