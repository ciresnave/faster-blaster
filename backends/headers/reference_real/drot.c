/**
 * @file drot.c
 * @brief Reference implementation of DROT (Double precision Givens rotation)
 *
 * DROT: apply plane rotation
 *   x_i := c * x_i + s * y_i
 *   y_i := c * y_i - s * x_i
 */

#include <blas_reference.h>

/**
 * Double precision apply plane rotation
 */
void drot_ref(int n, double *x, int incx, double *y, int incy, double c, double s) {
    if (n <= 0) return;
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            double temp = c * x[i] + s * y[i];
            y[i] = c * y[i] - s * x[i];
            x[i] = temp;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            double temp = c * x[ix] + s * y[iy];
            y[iy] = c * y[iy] - s * x[ix];
            x[ix] = temp;
            ix += incx;
            iy += incy;
        }
    }
}
