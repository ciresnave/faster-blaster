/**
 * @file dscal.c
 * @brief Reference implementation of DSCAL (Double precision scale vector)
 *
 * DSCAL: x := alpha * x
 */

#include <blas_reference.h>

/**
 * Double precision scale vector
 */
void dscal_ref(int n, double alpha, double *x, int incx) {
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
