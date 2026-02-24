/**
 * @file zcopy.c
 * @brief Reference implementation of ZCOPY (Complex double precision copy vector)
 *
 * ZCOPY: y := x
 */

#include <blas_reference.h>
#include <string.h>

typedef struct {
    double real;
    double imag;
} complex_d;

/**
 * Complex double precision copy vector
 */
void zcopy_ref(int n, const complex_d *x, int incx, complex_d *y, int incy) {
    if (n <= 0) return;
    
    if (incx == 1 && incy == 1) {
        memcpy(y, x, n * sizeof(complex_d));
    } else if (incx == incy && incx == -1) {
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
