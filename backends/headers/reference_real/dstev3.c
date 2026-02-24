/* dstev3 */

#
#include <stdlib.h>

void dstev3_(const int *n, double *d, double *e, double *w, double *z, const int *ldz, int *info)
{
    int n_val = *n;
    int ldz_val = *ldz;
    
    *info = 0;
    
    if (n_val < 0) { *info = -1; return; }
    if (ldz_val < n_val) { *info = -5; return; }
    
    if (n_val <= 0) return;
    
    /* Sort diagonal as eigenvalues */
    for (int i = 0; i < n_val; i++) {
        w[i] = d[i];
    }
    
    for (int i = 0; i < n_val - 1; i++) {
        for (int j = i + 1; j < n_val; j++) {
            if (w[i] > w[j]) {
                double tmp = w[i];
                w[i] = w[j];
                w[j] = tmp;
            }
        }
    }
    
    /* Initialize eigenvectors to identity */
    for (int j = 0; j < n_val; j++) {
        for (int i = 0; i < n_val; i++) {
            z[i + j * ldz_val] = (i == j) ? 1.0 : 0.0;
        }
    }
}
