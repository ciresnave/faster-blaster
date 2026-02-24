/**
 * @file drotg.c
 * @brief Reference implementation of DROTG (Double precision Givens rotation generator)
 *
 * DROTG: compute plane rotation parameters (double precision)
 */

#include <blas_reference.h>
#

/**
 * Double precision Givens rotation generator
 */
void drotg_ref(double *a, double *b, double *c, double *s) {
    if (a == NULL || b == NULL || c == NULL || s == NULL) return;
    
    double f = *a;
    double g = *b;
    double h;
    
    if (fabs(g) < 1e-15) {
        *c = 1.0;
        *s = 0.0;
        h = f;
    } else if (fabs(f) < 1e-15) {
        *c = 0.0;
        *s = (g >= 0.0) ? 1.0 : -1.0;
        h = fabs(g);
    } else {
        h = sqrt(f * f + g * g);
        if (h < 0.0) h = -h;
        *c = f / h;
        *s = g / h;
    }
    
    *a = h;
}
