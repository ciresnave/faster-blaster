/**
 * @file izamax.c
 * @brief Reference implementation of IZAMAX (Complex double precision index of maximum absolute value)
 *
 * IZAMAX: result = index i where |x_i| is maximum (absolute value of complex number)
 *
 * Returns 1-based index for BLAS compatibility.
 */

#include <blas_reference.h>
#

typedef struct {
    double real;
    double imag;
} complex_d;

/**
 * Complex double precision index of maximum absolute value (1-based indexing)
 */
int izamax_ref(int n, const complex_d *x, int incx) {
    if (n <= 0) return 0;
    if (incx <= 0) return 0;
    
    int max_idx = 1;
    double max_val = sqrt(x[0].real * x[0].real + x[0].imag * x[0].imag);
    
    int ix = incx;
    for (int i = 2; i <= n; i++) {
        double absval = sqrt(x[ix].real * x[ix].real + x[ix].imag * x[ix].imag);
        if (absval > max_val) {
            max_val = absval;
            max_idx = i;
        }
        ix += incx;
    }
    
    return max_idx;
}
