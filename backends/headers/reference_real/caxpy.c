/**
 * @file caxpy.c
 * @brief Reference implementation of CAXPY (Complex single precision AXPy)
 *
 * CAXPY: y := alpha*x + y (complex single precision)
 */

#include <blas_reference.h>

/* Complex number type for single precision */
typedef struct {
    float real;
    float imag;
} complex_f;

/**
 * Complex single precision multiplication
 */
static inline complex_f cmul_f(complex_f a, complex_f b) {
    return (complex_f){
        a.real * b.real - a.imag * b.imag,
        a.real * b.imag + a.imag * b.real
    };
}

/**
 * Complex single precision vector y := alpha*x + y
 */
void caxpy_ref(int n, complex_f alpha, const complex_f *x, int incx, complex_f *y, int incy) {
    if (n <= 0) return;
    if (alpha.real == 0.0f && alpha.imag == 0.0f) return;
    
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; i++) {
            complex_f prod = cmul_f(alpha, x[i]);
            y[i].real += prod.real;
            y[i].imag += prod.imag;
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        int iy = (incy > 0) ? 0 : (n - 1) * (-incy);
        
        for (int i = 0; i < n; i++) {
            complex_f prod = cmul_f(alpha, x[ix]);
            y[iy].real += prod.real;
            y[iy].imag += prod.imag;
            ix += incx;
            iy += incy;
        }
    }
}
