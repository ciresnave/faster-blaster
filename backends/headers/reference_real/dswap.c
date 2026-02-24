/**
 * @file dswap.c
 * @brief Reference implementation of DSWAP (Double precision swap vectors)
 *
 * DSWAP: swap x and y
 */

#include <blas_reference.h>

/**
 * Double precision swap vectors
 */
void dswap_ref(int n, double *x, int incx, double *y, int incy) {
    if (n <= 0) return;
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            double tmp = x[i];
            x[i] = y[i];
            y[i] = tmp;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            double tmp = x[ix];
            x[ix] = y[iy];
            y[iy] = tmp;
            ix += incx;
            iy += incy;
        }
    }
}
