/**
 * @file srot.c
 * @brief Reference implementation of SROT (Single precision Givens rotation)
 *
 * SROT: apply plane rotation
 *   x_i := c * x_i + s * y_i
 *   y_i := c * y_i - s * x_i
 */

#include <blas_reference.h>

/**
 * Single precision apply plane rotation
 */
void srot_ref(int n, float *x, int incx, float *y, int incy, float c, float s) {
    if (n <= 0) return;
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            float temp = c * x[i] + s * y[i];
            y[i] = c * y[i] - s * x[i];
            x[i] = temp;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            float temp = c * x[ix] + s * y[iy];
            y[iy] = c * y[iy] - s * x[ix];
            x[ix] = temp;
            ix += incx;
            iy += incy;
        }
    }
}
