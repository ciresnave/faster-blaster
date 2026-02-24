/**
 * @file dzasum.c
 * @brief Reference implementation of DZASUM (Complex double precision absolute value sum)
 *
 * DZASUM: result = sum(|real(x_i)| + |imag(x_i)|)
 */

#include <blas_reference.h>
#

typedef struct {
    double real;
    double imag;
} complex_d;

/**
 * Complex double precision sum of absolute values
 */
double dzasum_ref(int n, const complex_d *x, int incx) {
    if (n <= 0) return 0.0;
    if (incx <= 0) return 0.0;
    
    double sum = 0.0;
    int ix = 0;
    
    for (int i = 0; i < n; i++) {
        sum += fabs(x[ix].real) + fabs(x[ix].imag);
        ix += incx;
    }
    
    return sum;
}
