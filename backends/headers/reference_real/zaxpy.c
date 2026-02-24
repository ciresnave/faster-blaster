/**
 * @file zaxpy.c
 * @brief Reference implementation of ZAXPY (Complex double precision AXPy)
 *
 * ZAXPY: y := alpha*x + y (complex double precision)
 */

#include <blas_reference.h>

/* Complex number type for double precision */
typedef struct {
    double real;
    double imag;
} complex_d;

/**
 * Complex double precision multiplication
 */
static inline complex_d cmul_d(complex_d a, complex_d b) {
    return (complex_d){
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

/**
 * Complex double precision vector y := alpha*x + y
 */
void zaxpy_ref(int n, complex_d alpha, const complex_d *x, int incx, complex_d *y, int incy) {
    if (n <= 0) return;
    if (alpha.real == 0.0 && alpha.imag == 0.0) return;
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            complex_d prod = cmul_d(alpha, x[i]);
            y[i].real += prod.real;
            y[i].imag += prod.imag;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            complex_d prod = cmul_d(alpha, x[ix]);
            y[iy].real += prod.real;
            y[iy].imag += prod.imag;
            ix += incx;
            iy += incy;
        }
    }
}
