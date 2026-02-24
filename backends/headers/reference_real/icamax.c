/**
 * @file icamax.c
 * @brief Reference implementation of ICAMAX (Complex single precision index of maximum absolute value)
 *
 * ICAMAX: result = index i where |x_i| is maximum (absolute value of complex number)
 *
 * Returns 1-based index for BLAS compatibility.
 */

#include <blas_reference.h>
#

typedef struct {
    float real;
    float imag;
} complex_f;

/**
 * Complex single precision index of maximum absolute value (1-based indexing)
 */
int icamax_ref(int n, const complex_f *x, int incx) {
    if (n <= 0) return 0;
    if (incx <= 0) return 0;
    
    int max_idx = 1;
    float max_val = sqrtf(x[0].real * x[0].real + x[0].imag * x[0].imag);
    
    int ix = incx;
    for (int i = 2; i <= n; i++) {
        float absval = sqrtf(x[ix].real * x[ix].real + x[ix].imag * x[ix].imag);
        if (absval > max_val) {
            max_val = absval;
            max_idx = i;
        }
        ix += incx;
    }
    
    return max_idx;
}
