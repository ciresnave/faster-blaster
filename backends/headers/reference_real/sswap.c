/**
 * @file sswap.c
 * @brief Reference implementation of SSWAP (Single precision swap vectors)
 *
 * SSWAP: swap x and y
 */

#include <blas_reference.h>

/**
 * Single precision swap vectors
 */
void sswap_ref(int n, float *x, int incx, float *y, int incy) {
    if (n <= 0) return;
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            float tmp = x[i];
            x[i] = y[i];
            y[i] = tmp;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            float tmp = x[ix];
            x[ix] = y[iy];
            y[iy] = tmp;
            ix += incx;
            iy += incy;
        }
    }
}
