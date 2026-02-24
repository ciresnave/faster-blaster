/**
 * @file cscal.c
 * @brief Reference implementation of CSCAL (Complex single precision scale vector)
 *
 * CSCAL: x := alpha * x (complex)
 */

#include <blas_reference.h>

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
 * Complex single precision scale vector
 */
void cscal_ref(int n, complex_f alpha, complex_f *x, int incx) {
    if (n <= 0) return;
    
    if (incx == 1) {
        for (int i = 0; i < n; i++) {
            x[i] = cmul_f(alpha, x[i]);
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        for (int i = 0; i < n; i++) {
            x[ix] = cmul_f(alpha, x[ix]);
            ix += incx;
        }
    }
}
