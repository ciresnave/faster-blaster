/**
 * @file sscal.c
 * @brief Reference implementation of SSCAL (Single precision scale vector)
 *
 * SSCAL: x := alpha * x
 */

#include <blas_reference.h>

/**
 * Single precision scale vector
 */
void sscal_ref(int n, float alpha, float *x, int incx) {
    if (n <= 0) return;
    
    if (incx == 1) {
        for (int i = 0; i < n; i++) {
            x[i] *= alpha;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        for (int i = 0; i < n; i++) {
            x[ix] *= alpha;
            ix += incx;
        }
    }
}
