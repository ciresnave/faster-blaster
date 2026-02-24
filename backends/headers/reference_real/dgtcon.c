/* dgtcon */

#
#include <stdlib.h>

void dgtcon_(const char *norm, const int *n, const double *dl, const double *d,
             const double *du, const double *du2, const int *ipiv,
             const double *anorm, double *rcond, double *work, int *iwork, int *info)
{
    int n_val = *n;
    double anorm_val = *anorm;
    
    *info = 0;
    *rcond = 0.0;
    
    if (norm[0] != 'O' && norm[0] != '1' && norm[0] != 'I') { *info = -1; return; }
    if (n_val < 0) { *info = -2; return; }
    if (anorm_val < 0.0) { *info = -8; return; }
    
    if (n_val <= 0) {
        *rcond = 1.0;
        return;
    }
    
    /* Simplified: estimate reciprocal condition number */
    double rmin = fabs(d[0]);
    for (int i = 1; i < n_val; i++) {
        double di = fabs(d[i]);
        if (di < rmin) rmin = di;
    }
    
    if (rmin > 1e-12 && anorm_val > 0.0) {
        *rcond = rmin / anorm_val;
    } else {
        *rcond = 1e-15;
    }
}
