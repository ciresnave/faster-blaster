/* Tridiagonal subset */

#
#include <stdlib.h>

void dstevx_(const char *jobz, const char *range, const int *n, double *d, double *e,
             const double *vl, const double *vu, const int *il, const int *iu,
             const double *abstol, int *m, double *w, double *z, const int *ldz,
             double *work, int *iwork, int *ifail, int *info)
{
    int n_val = *n;
    int ldz_val = *ldz;
    double vl_val = *vl;
    double vu_val = *vu;
    
    *info = 0;
    *m = 0;
    
    if (jobz[0] != 'N' && jobz[0] != 'V') { *info = -1; return; }
    if (range[0] != 'A' && range[0] != 'V' && range[0] != 'I') { *info = -2; return; }
    if (n_val < 0) { *info = -3; return; }
    
    if (n_val <= 0) return;
    
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
    
    if (range[0] == 'A') {
        *m = n_val;
    } else if (range[0] == 'V') {
        int count = 0;
        for (int i = 0; i < n_val; i++) {
            if (w[i] >= vl_val && w[i] <= vu_val) {
                w[count] = w[i];
                count++;
            }
        }
        *m = count;
    } else {
        *m = iu[0] - il[0] + 1;
        for (int i = 0; i < *m; i++) {
            w[i] = w[il[0] - 1 + i];
        }
    }
    
    if (jobz[0] == 'V') {
        for (int j = 0; j < *m; j++) {
            for (int i = 0; i < n_val; i++) {
                z[i + j * ldz_val] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
}
