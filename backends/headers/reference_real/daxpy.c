/**
 * @file daxpy.c
 * @brief Reference implementation of DAXPY (Double precision AXPy)
 *
 * DAXPY: y := alpha*x + y (double precision)
 */

#include <blas_reference.h>

/**
 * Double precision vector y := alpha*x + y
 */
void daxpy_ref(int n, double alpha, const double *x, int incx, double *y, int incy) {
    if (n <= 0) return;
    if (alpha == 0.0) return;
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            y[i] += alpha * x[i];
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            y[iy] += alpha * x[ix];
            ix += incx;
            iy += incy;
        }
    }
}
