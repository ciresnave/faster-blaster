/**
 * @file scasum.c
 * @brief Reference implementation of SCASUM (Complex single precision absolute value sum)
 *
 * SCASUM: result = sum(|real(x_i)| + |imag(x_i)|)
 */

#include <blas_reference.h>
#

typedef struct {
    float real;
    float imag;
} complex_f;

/**
 * Complex single precision sum of absolute values
 */
float scasum_ref(int n, const complex_f *x, int incx) {
    if (n <= 0) return 0.0f;
    if (incx <= 0) return 0.0f;
    
    float sum = 0.0f;
    int ix = 0;
    
    for (int i = 0; i < n; i++) {
        sum += fabsf(x[ix].real) + fabsf(x[ix].imag);
        ix += incx;
    }
    
    return sum;
}
