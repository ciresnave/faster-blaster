/* sgtcon */

#
#include <stdlib.h>

void sgtcon_(const char *norm, const int *n, const float *dl, const float *d,
             const float *du, const float *du2, const int *ipiv,
             const float *anorm, float *rcond, float *work, int *iwork, int *info)
{
    int n_val = *n;
    float anorm_val = *anorm;
    
    *info = 0;
    *rcond = 0.0f;
    
    if (norm[0] != 'O' && norm[0] != '1' && norm[0] != 'I') { *info = -1; return; }
    if (n_val < 0) { *info = -2; return; }
    if (anorm_val < 0.0f) { *info = -8; return; }
    
    if (n_val <= 0) {
        *rcond = 1.0f;
        return;
    }
    
    /* Simplified: estimate reciprocal condition number */
    float rmin = fabsf(d[0]);
    for (int i = 1; i < n_val; i++) {
        float di = fabsf(d[i]);
        if (di < rmin) rmin = di;
    }
    
    if (rmin > 1e-6f && anorm_val > 0.0f) {
        *rcond = rmin / anorm_val;
    } else {
        *rcond = 1e-10f;
    }
}
