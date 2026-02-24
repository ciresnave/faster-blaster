/**
 * @file zscal.c
 * @brief Reference implementation of ZSCAL (Complex double precision scale vector)
 *
 * ZSCAL: x := alpha * x (complex)
 */

#include <blas_reference.h>

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
 * Complex double precision scale vector
 */
void zscal_ref(int n, complex_d alpha, complex_d *x, int incx) {
    if (n <= 0) return;
    
    if (incx == 1) {
        for (int i = 0; i < n; i++) {
            x[i] = cmul_d(alpha, x[i]);
        }
    } else {
        int ix = (incx > 0) ? 0 : (n - 1) * (-incx);
        for (int i = 0; i < n; i++) {
            x[ix] = cmul_d(alpha, x[ix]);
            ix += incx;
        }
    }
}
