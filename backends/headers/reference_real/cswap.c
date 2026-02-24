/**
 * @file cswap.c
 * @brief Reference implementation of CSWAP (Complex single precision swap vectors)
 *
 * CSWAP: swap x and y
 */

#include <blas_reference.h>

typedef struct {
    float real;
    float imag;
} complex_f;

/**
 * Complex single precision swap vectors
 */
void cswap_ref(int n, complex_f *x, int incx, complex_f *y, int incy) {
    if (n <= 0) return;
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            complex_f tmp = x[i];
            x[i] = y[i];
            y[i] = tmp;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            complex_f tmp = x[ix];
            x[ix] = y[iy];
            y[iy] = tmp;
            ix += incx;
            iy += incy;
        }
    }
}
