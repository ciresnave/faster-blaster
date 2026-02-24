/**
 * @file scopy.c
 * @brief Reference implementation of SCOPY (Single precision copy vector)
 *
 * SCOPY: y := x (copy vector)
 */

#include <blas_reference.h>
#include <string.h>

/**
 * Single precision copy vector
 */
void scopy_ref(int n, const float *x, int incx, float *y, int incy) {
    if (n <= 0) return;
    
    if (incx == 1 && incy == 1) {
        /* Contiguous: use memcpy for efficiency */
        memcpy(y, x, n * sizeof(float));
    } else if (incx == incy && incx == -1) {
        /* Handle backward copy for overlapping arrays */
        int ix = (n - 1);
        int iy = (n - 1);
        for (int i = 0; i < n; i++) {
            y[iy] = x[ix];
            ix--;
            iy--;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            y[iy] = x[ix];
            ix += incx;
            iy += incy;
        }
    }
}
