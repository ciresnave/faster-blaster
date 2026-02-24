/**
 * @file srotg.c
 * @brief Reference implementation of SROTG (Single precision Givens rotation generator)
 *
 * SROTG: compute plane rotation parameters
 *
 * Given a and b, compute c, s, r, z such that:
 *   c * a + s * b = r
 *   -s * a + c * b = 0
 *
 * Where c^2 + s^2 = 1
 */

#include <blas_reference.h>
#

/**
 * Single precision Givens rotation generator
 */
void srotg_ref(float *a, float *b, float *c, float *s) {
    if (a == NULL || b == NULL || c == NULL || s == NULL) return;
    
    float f = *a;
    float g = *b;
    float h;
    
    if (fabsf(g) < 1e-10f) {
        *c = 1.0f;
        *s = 0.0f;
        h = f;
    } else if (fabsf(f) < 1e-10f) {
        *c = 0.0f;
        *s = (g >= 0.0f) ? 1.0f : -1.0f;
        h = fabsf(g);
    } else {
        h = sqrtf(f * f + g * g);
        if (h < 0.0f) h = -h;
        *c = f / h;
        *s = g / h;
    }
    
    *a = h;
}
