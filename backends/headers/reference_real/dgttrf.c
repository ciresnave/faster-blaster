#
#include <string.h>
#include <blas_reference.h>

/* Tridiagonal factorization with partial pivoting */
void dgttrf_(const int *n, double *dl, double *d, double *du, double *du2,
              int *ipiv, int *info)
{
    int n_val = *n;
    *info = 0;

    if (n_val <= 0) return;

    /* Gaussian elimination with partial pivoting */
    for (int i = 0; i < n_val; i++) {
        ipiv[i] = i + 1;  /* 1-based indexing */

        if (i + 1 < n_val && fabs(dl[i]) > fabs(d[i])) {
            /* Swap rows i and i+1 */
            double tmp_d = d[i];
            d[i] = dl[i];
            dl[i] = tmp_d;
            ipiv[i] = i + 2;

            if (i + 1 < n_val - 1) {
                du2[i] = du[i];
                du[i] = du[i + 1];
            }
        }

        /* Check for zero pivot */
        if (d[i] == 0.0) {
            *info = i + 1;
            return;
        }

        /* Scale subdiagonal */
        if (i + 1 < n_val) {
            dl[i] /= d[i];
            d[i + 1] -= dl[i] * du[i];
        }
    }
}
